---
version: v0.1
audience: cc / 工程师
purpose: F1 Attribution Ledger Phase A 的 SQLite 真源与运维方式
status: 试运行
---

# Ledger Phase A

| RULE-ID | 触发 | 要求 | Why |
|---------|------|------|-----|
| LEDGER-SQLITE-SOURCE | online fit 成功后写 lineage | 先 atomic ckpt，再事务写 SQLite ledger | ckpt 是主路径真相，ledger 是治理索引，不允许反过来阻塞训练 |
| LEDGER-READ-ONLY-QUERY | CLI / 排障 / 审计 | 读取统一走只读 SQLite URI | 避免排障工具误写真源 |
| LEDGER-CONSISTENCY-SCAN | 断电 / 崩溃 / schema 怀疑漂移 | 跑 `consistency` 扫描 4 类 dirty 指标 | Phase A 允许发现 dirty，不允许 silent corruption |

## Schema

Phase A 落 3 张业务表 + 1 张元数据表：

| 表 | 主键 | 作用 |
|----|------|------|
| `fit_runs` | `fit_run_id` | 记录 `ckpt_in_sha8 -> ckpt_out_sha8`、batch 大小、config 快照 |
| `event_attribution` | `(fit_run_id, event_uuid)` | 记录该次 fit 吃到的 event membership、batch 顺序、窗口帧 |
| `delta_summary` | `fit_run_id` | 记录整体 `tau_drift` 变化摘要，不做 per-event dense attribution |
| `schema_meta` | `key` | 当前仅存 `version=ledger_v1` |

必要索引：

- `fit_runs(ckpt_out_sha8)` UNIQUE
- `fit_runs(ckpt_in_sha8)`
- `event_attribution(event_uuid)`

**Why**：Phase A 的核心查询是 `ckpt_out -> fit_run -> events -> ckpt_in` 反查链，以及 `ckpt_in -> child fit_runs` 的 1:N 展开。

## Fail-Safe 顺序

```text
fit_step
  -> atomic tmp ckpt write
  -> os.replace(final ckpt)
  -> compute ckpt_out_sha8
  -> BEGIN IMMEDIATE
  -> insert fit_runs + event_attribution + delta_summary
  -> COMMIT
```

若 ledger 写入失败：

- 训练主路径不回滚已落盘 ckpt
- `try_fit_step()` 仅 `warning`
- 返回结果里标记 `metadata["ledger_write_failed"] = True`

**Why**：主路径 first，ledger second，符合 ADR-038 的 fail-safe 顺序，避免治理基础设施反向拖死训练。

## CLI

默认路径：`~/.lnmem/ledger/ledger.sqlite`

```bash
python -m poc.src.lnn.ledger.cli lineage ef678901
python -m poc.src.lnn.ledger.cli --db /tmp/ledger.sqlite list --limit 5
python -m poc.src.lnn.ledger.cli consistency --json
python -m poc.src.lnn.ledger.cli inspect <fit_run_id>
```

`lineage` 人类可读输出示例：

```text
fit_run_id: 3d3b...
ckpt_in_sha8 → ckpt_out_sha8: abc12345 → ef678901
fired_at: 2026-05-28T10:00:00Z
persona_idx: 30
events (8): ['event-1', 'event-2']
delta_l2_total: 0.0123
```

**Why**：Phase A 排障首先是人工确认 lineage 链是否闭合，CLI 必须直接可读，`--json` 再服务脚本。

## 故障排查

### schema mismatch

- 现象：`ledger schema mismatch: expected ledger_v1`
- 动作：停写，先迁移或重建；不要让新代码混写旧 schema

### corrupt DB

- 现象：reader 抛 `LedgerReadError` 或 SQLite `database disk image is malformed`
- 动作：先备份原文件，再用 `sqlite3 ledger.sqlite 'PRAGMA integrity_check;'`

### lock contention

- 现象：写入卡在 `BEGIN IMMEDIATE` 或报 `database is locked`
- 动作：确认是否有长事务；Phase A 已启 WAL + `busy_timeout=5000`，持续冲突再排查外部进程

### dirty report

- 现象：`consistency` 返回 `runs_without_events` / `orphan_events` 非 0
- 动作：优先视为崩溃残留或人工注入测试数据，按 `fit_run_id` 用 `inspect` 定位

**Why**：Phase A 允许 dirty 被发现和解释，不允许“看起来没报错”就当账本健康。
