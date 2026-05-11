# M0 W2 P0-1: daemon V0.3 设计 — user_triggered_mismatch channel + trigger 平衡

> 状态：cc 修订 V2（响应 codex 一审 BLOCK 9 findings）→ 待川哥签 / 可选二审
> 作者：cc | 2026-05-11 09:00 CST 初稿 / 09:30 V2 修订
> 范围：daemon trigger 子系统 + 反馈 schema + CLI 客户端
> 一审报告：`/tmp/codex-v03-review-output.md`
>
> **V2 关键修订**：F1 reward 语义统一 / F2 frame 对齐契约硬化 / F3 schema 加时间字段 /
> F6 加 Header token 鉴权 + 独立 rate-limit / F8 测试加 6 类边界 / F4+F9 拆两阶段实现

---

## A. 问题陈述

**根因数据**（PoC Day 9-14 累积，2026-05-04 至 2026-05-11）：

| label | 累积 | 占比 |
|-------|------|------|
| match | 12 | 46% |
| skip | 10 | 38% |
| input | 3 | 12% |
| **mismatch** | **1** | **4%** |
| **总** | **26** | 100% |

**痛点链**：

1. daemon V0.2 trigger 设计**偏 low_vlm_conf 路径**（"系统自己拿不准 → 弹窗问"）：26 次 trigger 中 low_vlm_conf 占 61%
2. 川哥对 low_vlm_conf 类弹窗**主要 skip 不回应**（弹窗提示"皱眉=0.00 笑=1.00 ..."，川哥感知不到自己当下被分类成什么状态，无从对错）
3. brow_burst 类 trigger（阈值 > 0.85）触发**都是高强度皱眉**，川哥认真回应但**几乎都是 match**（"是的我在思考"），少有 mismatch
4. **结果**：mismatch 6 天累积仅 1 条，远低于 [online_trainer.py:45](poc/src/lnn/online_trainer.py:45) `trigger_if_ready` 启动门槛 `min_mismatch_for_fit ≥ 10`
5. **Phase 2 在线学习闭环工程链 100% 实现但永远启动不了** — daemon V0.2 没有让川哥**主动告诉模型"你刚才判错了"**的通道

**核心命题**：现有 trigger 模式假设"模型不确定时问用户" → 但用户对模型确定的错判更有动力纠正；缺这条通道，反馈数据偏向收集"系统知道自己不确定"，不收集"系统不知道自己错了"，正是 mismatch 短缺的根因。

### A.bis 统计口径锁定（响应 codex F5）

**问题**：历史文档 11/17/19/26 多版本（poc-final-report.md:142 / day14-progress.md:208 / handoff-day15.md:35 / 本设计 §A 表 26），未给统一口径复算证据。

**口径定义**：本设计 §A 表"26 条"= `~/.lnmem/feedback/queries_2026-05-*.jsonl` 全量行数（不含 skipped_*.jsonl），label 分布按每行 `user_response.label` 计数。

**复算命令**（供 reviewer 复现）：
```bash
cat ~/.lnmem/feedback/queries_2026-05-*.jsonl | python3 -c "
import sys, json
from collections import Counter
labels = Counter()
for line in sys.stdin: labels[json.loads(line).get('user_response',{}).get('label','?')] += 1
print('total:', sum(labels.values()), 'dist:', dict(labels))
"
```

**口径漂移说明**：
- "11 条" = PoC Day 9 daemon /health responses 字段（早期未同步 jsonl）
- "17 条" = PoC Day 14 Loop 13 重新校准（首次按 jsonl 复算）
- "19 条" = HANDOFF 21:30 时点累积
- "26 条" = 本设计 09:00 时点（夜间 +7 反馈：6 match + 1 input）

**混杂项验证 todo**（非本 P0-1 阻塞，移到 P0-3 vboss_self 阶段）：分 vboss_001 baseline 期（Day 9-11，未入 vboss_self 期）vs vboss_self 入 ckpt 后期（Day 12+）两段 mismatch 率，排除 persona 基线贴合度作为混杂项。

---

## B. 设计目标

1. **新增 user_triggered_mismatch channel**：让川哥**主动**告知系统当前状态判断错误（不等系统问）
2. **trigger 阈值平衡**：减少 low_vlm_conf 类过度触发，多样化 brow_burst 之外的被动 trigger
3. **反馈 schema 向后兼容**：V0.3 是叠加而非替换；V0.2 累积的 26 条反馈仍可用作 baseline
4. **LNN-FIRST P-1 合规**：新 channel 写入相同 JSONL 格式，被同一份 [feedback_to_batch.py](poc/src/lnn/feedback_to_batch.py) 消费，下游 fit_step 无改动
5. **可观测**：新加 trigger 类型分布 metric，写入 daemon `/health` 让川哥实时看分布

---

## C. 设计方案

### C.1 新增 trigger 类型：`user_triggered`

| 属性 | 值 |
|------|---|
| trigger 主体 | 用户主动（不依赖 daemon 信号判断） |
| 写入字段 | `reason: "user_triggered"`，`signal: null`（**仅审计用，不参与训练**） |
| user_response.label | `mismatch`（默认）或 `input`（用户提供正确 state） |
| **reward 取值** | **按 [REWARD_MAP](../poc/src/lnn/online_contracts.py:33) 走：label=mismatch → -1.0；label=input → 0.0。禁止本地 override**（响应 codex F1） |
| true_state | 必填（用户输入的真实状态） |
| pred_state | 当前 daemon 推理结果（daemon 写） |
| note | 选填（用户自由文字） |
| event_uuid | 必填（UUID4） |
| session_id | 当前 daemon session |
| **frame_id** | **必填**：(a) 用户在 POST 时指认（优选） (b) fallback = daemon 最近一帧（响应 codex F3） |
| **trigger_ref_frame_ts** | **新增必填**：frame_id 对应帧的真实摄入时间戳（ISO8601）（响应 codex F3） |
| **client_action_ts** | **新增必填**：用户点击/输入的时间戳（CLI 或 daemon 接收 POST 时打）（响应 codex F3） |
| **alignment_latency_s** | **新增计算字段**：`client_action_ts - trigger_ref_frame_ts`（响应 codex F3） |
| **caller_source** | **新增审计字段**：`{pid, cmdline_first_arg}`（响应 codex F6 审计） |

**示例 JSONL 行（V2，含响应 codex F3+F6 加固字段）**：
```json
{
  "ts": "2026-05-11T09:30:00.000000Z",
  "frame_id": 35000,
  "trigger_ref_frame_ts": "2026-05-11T09:29:58.000000Z",
  "client_action_ts": "2026-05-11T09:30:00.000000Z",
  "alignment_latency_s": 2.0,
  "reason": "user_triggered",
  "signal": null,
  "vlm_label": null,
  "user_response": {
    "label": "mismatch",
    "true_state": "engaged_focused",
    "note": "我在认真写文档不是分心"
  },
  "latency_s": 0.0,
  "session_id": "<current>",
  "event_uuid": "<uuid4>",
  "caller_source": {"pid": 88123, "cmdline": "lnmem-tell"}
}
```

### C.2 trigger 阈值平衡（响应 codex F4 — 拆 Phase A/B）

**V0.3 Phase A（本 P0-1 范围，无离线证据情况下零调整 V0.2）**：

| trigger 类型 | V0.2 阈值（保持） | 备注 |
|------------|----------|---------|
| low_vlm_conf | confidence < 0.65 | **不变**（无离线回放证据前不调） |
| brow_burst | brow_furrow > 0.85 ∧ window=10 | **不变**（V0.2 收紧已经过 22/4min 过敏实测） |

**V0.3 Phase B（独立 P0-1.5 子任务，先做离线回放再上线，本设计不实施）**：

阈值调参与新 trigger 类型（smile_burst / pose_held / brow_burst 阈值放宽）拆分到 P0-1.5：
1. 用 `~/.lnmem/data/signals_*.jsonl` 全量回放，比较新旧阈值的 (a) 每小时触发数 (b) 触发分布
2. 验证不超过 V0.2 baseline 30/h 弹窗上限
3. 川哥 sign-off 后再上线，独立 commit

**rate-limit / DnD 不变**：现有 [persistence.py](../poc/src/realtime/feedback/persistence.py) 限速 + DnD 窗口配置保留。

**user_triggered 独立 rate-limit**（响应 codex F6）：1 次 / 分钟（防恶意进程批量投毒触发 fit_step）。

### C.3 数据接口

#### C.3.1 daemon HTTP 端点（新增，V2 含 F3+F6 加固）

**POST /trigger_mismatch**

**Header（必填，响应 codex F6）**：
- `X-LnMem-Token: <secret>` — 共享 token，daemon 启动时从 `~/.lnmem/secret_token` 读（chmod 600）
- 缺 / 错 → 401

请求 body（JSON，V2）：
```json
{
  "true_state": "engaged_focused",      // required
  "note": "我在写文档",                  // optional
  "label": "mismatch",                  // optional, default="mismatch", enum=["mismatch","input"]
  "frame_id": 35000,                    // optional, 用户指认；缺则 fallback "最近帧"（响应 codex F3）
  "client_action_ts": "2026-05-11T09:30:00.000000Z"  // optional, ISO8601，缺则 daemon 收到 POST 时打
}
```

响应（V2）：
```json
{
  "ok": true,
  "event_uuid": "<uuid4>",
  "frame_id": 35000,
  "trigger_ref_frame_ts": "2026-05-11T09:29:58.000000Z",
  "pred_state": "low_energy",
  "alignment_latency_s": 2.0,
  "warning": null                       // 如 alignment_latency_s > 5s 给警告但仍写入
}
```

错误响应（4xx/5xx）：
- 400: true_state 不在合法枚举内
- 400: label 不在 {"mismatch", "input"}
- 401: X-LnMem-Token 缺或错（响应 codex F6）
- 429: user_triggered rate-limit 触发（> 1 次/分钟）（响应 codex F6）
- 503: daemon 未推理过任何帧（frame_id 为 0 且未指定）
- 503: 指定的 frame_id 不在 daemon 最近 window 范围（无法 ref）

**对齐硬过滤（响应 codex F3）**：
- daemon 计算 `alignment_latency_s = client_action_ts - trigger_ref_frame_ts`
- 如果 > 5s：返回 200 但 response.warning 提示用户，JSONL 仍写入 `alignment_latency_s`，但 `feedback_to_batch` 加载时硬过滤掉 `> 5s` 的样本不入训练批（默认配置；可通过 `OnlineAdapterConfig.alignment_window_s` 调整）

#### C.3.2 CLI 客户端（新增，V2 含 token 读取）

文件：`scripts/lnmem-tell`（bash 简单 wrapper）

用法：
```bash
lnmem-tell <true_state> [note]
# 例：
lnmem-tell engaged_focused "我在写文档"
lnmem-tell tired
```

实现：
1. 从 `~/.lnmem/secret_token` 读 token（要求 chmod 600，否则报错退出）
2. 打 `client_action_ts = $(date -u +'%Y-%m-%dT%H:%M:%S.000000Z')`
3. curl POST + `-H "X-LnMem-Token: $TOKEN"` 到 daemon /trigger_mismatch
4. pretty print 响应，如果 `warning != null` 红字提示用户

**安全前置**：`lnmem-tell` 启动时 check `~/.lnmem/secret_token` 权限 ≤ 600，否则拒绝运行（防 token 泄漏给其他用户）。

#### C.3.3 反馈 schema 向后兼容（V2 — 响应 codex F2 重写）

- 现有 [persistence.py:8](../poc/src/realtime/feedback/persistence.py:8) reason 类型增至 3 个：`brow_burst | low_vlm_conf | user_triggered`
- queries_YYYY-MM-DD.jsonl 文件格式向后兼容（新字段 `trigger_ref_frame_ts` / `client_action_ts` / `alignment_latency_s` / `caller_source` 是 schema 增量字段）
- **feedback_to_batch.py user_triggered 处理路径**（修正 V1 错误）：
  - signal 字段：仅审计用，**不参与训练输入**（训练输入仍走 `(session_id, frame_id)` 切 signals_*.jsonl window）
  - frame_id 必须能在 signals_*.jsonl 找到对应帧 → 走标准 `_slice_window` 路径
  - 如果 frame_id 不在 signals 范围或 `alignment_latency_s > 5s` → **硬过滤掉，不入训练批**（与 V0.2 现有 skip 路径一致）
  - 新增 [`feedback_to_batch.py`](../poc/src/lnn/feedback_to_batch.py) 指标：
    - `user_triggered_total` — 收到的 user_triggered 反馈总数
    - `user_triggered_assembled` — 成功组装进训练批的数量
    - `user_triggered_skipped_alignment` — 因 alignment > 5s 跳过
    - `user_triggered_skipped_no_signal` — 因 signals 找不到 frame 跳过

#### C.3.4 trigger 类型分布 metric（V2 — 响应 codex F7 口径统一）

**新增 /health 字段**（口径：persistence.observe() 入口统一埋点，user_triggered 也走 persistence 路径）：

```json
{
  "feedback": {
    "observed": 39590,
    "triggered": 203,
    "rate_blocked": 179,
    "dnd_blocked": 0,
    "queried": 24,
    "responses": 24,
    "trigger_types": {                  // ← 新增
      "brow_burst": 7,
      "low_vlm_conf": 11,
      "user_triggered": 0
    },
    "user_triggered_metrics": {         // ← 新增
      "total": 0,
      "assembled": 0,
      "skipped_alignment": 0,
      "skipped_no_signal": 0
    }
  }
}
```

**口径关系明确化**（响应 codex F7）：
- `trigger_types.X` 与 `triggered` 总和：`sum(trigger_types) == queried`（POST /trigger_mismatch 也算 query）
- `triggered = queried + rate_blocked + dnd_blocked`
- `user_triggered_metrics.total` ⊆ `trigger_types.user_triggered`（前者细化训练入批结果）

**watchdog 兼容性**（响应 codex F7）：现有 [daemon_watchdog.sh](../scripts/daemon_watchdog.sh) 仅读 `/health.last_frame_ts` 单字段，新增的 `feedback.trigger_types` / `feedback.user_triggered_metrics` 是 sibling 字段不破解析路径。

---

## D. 实现方案

### D.1 daemon 改动点（V2 — Phase A 范围，响应 codex F4+F9）

**Phase A**（本 P0-1，仅 user_triggered 通道 + 鉴权 + 对齐 + 测试）：

| 文件 | 改动 | 行数估 |
|------|------|--------|
| `poc/src/realtime/feedback/persistence.py` | reason 枚举扩展 user_triggered + trigger_types metric + user_triggered_metrics + 新 schema 字段 | +40 |
| `poc/src/realtime/daemon_main.py` | POST /trigger_mismatch handler + Header token 鉴权 + rate-limit + secret_token 加载 + /health 新字段 | +120 |
| `poc/src/realtime/feedback/engine.py` | (无改动) | 0 |
| `poc/src/lnn/feedback_to_batch.py` | user_triggered 反馈处理（强 frame_id 对齐 + alignment_latency 硬过滤 + 4 指标） | +50 |
| `poc/src/lnn/online_contracts.py` | OnlineAdapterConfig 加 `alignment_window_s: float = 5.0` 字段 | +5 |
| `scripts/lnmem-tell` | CLI wrapper + token 权限检查 | +50 |
| `scripts/lnmem-init-token.sh` | 一键生成 secret_token（chmod 600） | +15 |

**Phase A 总估**：~280 行新增。**不在 Phase A 做**：trigger 阈值调整 / smile_burst / pose_held → 移到 Phase B（P0-1.5 独立任务）。

### D.2 新文件清单

```
poc/src/realtime/feedback/test_trigger_user_triggered.py   # 单测（新）
poc/src/realtime/test_daemon_trigger_mismatch.py            # 集成测（新）
scripts/lnmem-tell                                          # CLI（新）
docs/m0-w2-p0-1-daemon-v03-design.md                        # 本文档
```

### D.3 测试方案（V2 — 18 case，响应 codex F8 加 6 类边界）

**单测 test_trigger_user_triggered.py**（6 case）：
1. POST /trigger_mismatch 正常 → 200 + 写 JSONL + 含 trigger_ref_frame_ts / client_action_ts / alignment_latency_s
2. POST 缺 true_state → 400
3. POST true_state 不合法枚举 → 400
4. POST label=input → JSONL 写入 label=input 不是 mismatch
5. POST 缺 X-LnMem-Token → 401（响应 F6+F8）
6. POST 错 token → 401（响应 F6+F8）

**新增边界单测 test_trigger_edge_cases.py**（6 case — 响应 codex F8）：
7. **同一 frame_id 多次 user_triggered 在 60s 内提交** → 第 2 次起 429 rate-limit（去重 + rate-limit 双保险）
8. **并发 POST 同 frame_id**（threading 模拟 5 并发）→ 仅 1 写入成功，4 个 429（写文件加 fcntl.flock 单进程互斥）
9. **alignment_latency_s > 5s** → 返 200 + response.warning != null + JSONL 写入但 feedback_to_batch 后续过滤
10. **daemon 冷启动 frames_processed < window_size (10)** → 用户 POST 返 503（响应 F8）
11. **frame_id 不在 signals_*.jsonl 范围**（指认 frame_id=999999 不存在）→ 503（响应 F8）
12. **写文件失败模拟**（设 jsonl 文件 chmod 000）→ 5xx + 不写 partial 行（响应 F8 "失败清理"）

**集成测 test_daemon_trigger_mismatch.py**（3 case）：
13. CLI `lnmem-tell engaged_focused` → daemon → JSONL 出现 → reason=user_triggered + alignment_latency_s 字段齐备
14. user_triggered 反馈被 [feedback_to_batch.py](../poc/src/lnn/feedback_to_batch.py) 解析为 mismatch 进入训练批（前提：alignment_latency_s ≤ 5s + frame_id 在 signals 范围）
15. trigger_if_ready 在 user_triggered ≥ 10 mismatch 时返 True

**回归 test_v02_baseline.py 既有保护**（3 case）：
16. V0.2 既有 brow_burst 阈值 0.85 + window 10 行为不变（Phase A 不调阈值）
17. V0.2 既有 low_vlm_conf 阈值 0.65 行为不变（Phase A 不调阈值）
18. V0.2 累积 26 条反馈用 feedback_to_batch 加载仍 PASS（schema 向后兼容）

---

## E. 双门验证（实施后）

### E.1 Verify Gate（目标验证）

- [ ] 新加 10 个 test case 全 PASS
- [ ] 川哥试用 1 周累积 user_triggered mismatch ≥ 5（占新增反馈 ≥ 30%）
- [ ] trigger_types metric 显示分布更均匀（任一类型占比 < 70%）
- [ ] feedback_to_batch dry-run sanity 通过（user_triggered 反馈进训练批）

### E.2 Guard Gate（回归防护）

- [ ] V0.2 既有 373 测试全 PASS
- [ ] V0.2 累积 26 条反馈 schema 仍可被 feedback_to_batch 解析
- [ ] daemon `/health` 既有字段全保留
- [ ] watchdog W1-P2-1 修复后的脚本不受影响

---

## F. 时间盒（V2 — 响应 codex F9 拆 Phase A/B 重估）

### Phase A（本 P0-1 — user_triggered 通道 + 鉴权 + 对齐）

| 阶段 | V1 估时 | **V2 修正估时** | 负责 |
|------|--------|----------------|------|
| 设计文档（本文 V2） | 1h | 1.5h ✅ | cc |
| codex 对抗审一审 | 30-60min | ✅ 完成（9 finding，BLOCK） | codex (gpt-5.3-codex high) |
| 设计 V2 修订（响应 F1-F9） | - | **新增 1h** ✅ | cc |
| codex 二审（可选） | - | 30-45min | codex |
| 川哥签字 | 不定 | 不定 | 川哥 |
| **实现 Phase A** | 4-6h | **6-8h**（含鉴权 + 对齐 + 18 测试） | cc 主线 |
| Verify Gate 测试 | 1h | 1.5h（18 case） | cc |
| Guard Gate 回归 | 30min | 1h（V0.2 既有 373 测试） | cc |
| 川哥 1 周试用 | 7 天 async | 7 天 async | 川哥 |
| 验收 | 1h | 2h（含 metric / 数据 / 训练 dry-run） | cc |

**Phase A cc 主线总估**：**10-13h**（V1 估 7-9h 偏乐观，codex F9 命中）

### Phase B（P0-1.5 独立任务，本设计不实施）

| 阶段 | 估时 |
|------|------|
| 离线回放工具（signals_*.jsonl 全量重放） | 2h |
| 阈值 grid search + 触发数对比 | 2h |
| smile_burst / pose_held 实现 + 测试 | 3h |
| 川哥试用 1 周 | 7 天 |

**Phase B cc 主线总估**：~7-8h + 1 周观测。

---

## G. 不做的事（YAGNI，V2）

1. ❌ 不做桌面状态栏小工具（保 CLI 即可，桌面 UI 推迟到 M1）
2. ❌ 不做全局快捷键监听（macOS 全局快捷键需 Accessibility 授权，过度复杂）
3. ❌ **不做 reward 加权或 override**（响应 codex F1）：user_triggered 反馈 reward 严格按 [REWARD_MAP](../poc/src/lnn/online_contracts.py:33) 走，label=mismatch → -1.0，label=input → 0.0；禁止本地覆盖
4. ❌ 不改 reward_map（保持 [online_contracts.py](../poc/src/lnn/online_contracts.py) 现状）
5. ❌ 不接入 daemon V0.3 自学习闭环（那是 P0-3 范围；P0-1 只解决反馈通道）
6. ❌ **Phase A 不调 trigger 阈值**（响应 codex F4）：brow_burst / low_vlm_conf 保持 V0.2 已经过 22/4min 过敏实测的收紧值；阈值放宽 + 新 trigger 类型移到 Phase B（P0-1.5）独立任务
7. ❌ **Phase A 不做 smile_burst / pose_held**（响应 codex F4+F9）：移到 Phase B

---

## H. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| trigger 阈值放宽后过敏（弹窗 > 30/h） | 中 | 用户体验下降 | rate-limit 防线在；首日川哥手动监控弹窗频率，过敏立即回滚 |
| CLI 客户端使用门槛高（川哥不习惯命令行） | 中 | mismatch 累积仍慢 | 提供 1-key alias / 后续做桌面快捷键 |
| user_triggered 反馈的 frame_id 与 daemon 最近一帧错位（用户思考 → 输入 → 提交期间帧已往前） | 低 | 训练用错 fusion vector | frame_id 取**daemon 接收 POST 时**最近一帧，记录时间差 < 5s 才入训练批 |
| smile_burst / pose_held 新 trigger 类型 V0.3 首次见，调参不对 | 中 | 弹窗分布反而更偏 | 首周 metric 监控，> 70% 单类型占比 → 阈值微调 |

---

## I. 关联文档与依赖

- 上游设计：[`docs/poc-final-report.md`](poc-final-report.md) §C.3（PoC 反馈类型分布异常发现）
- 上游设计：[`docs/m0-w2-tasks.md`](m0-w2-tasks.md) P0-1 任务定义
- 下游依赖：M0 W2 P0-2（feedback v2 schema 加固）依赖本 P0-1 reason 枚举扩展
- 下游依赖：M0 W2 P0-3（vboss_self 微调）依赖本 P0-1 mismatch 累积 ≥ 10
- 已有实现：[`online_trainer.py`](../poc/src/lnn/online_trainer.py) / [`online_adapter.py`](../poc/src/lnn/online_adapter.py) / [`feedback_to_batch.py`](../poc/src/lnn/feedback_to_batch.py)（PoC 已完成）

---

## J. codex 对抗审 prompt（cc 派发用）

```
你的角色：对抗审查者。审查 LnMem M0 W2 P0-1 daemon V0.3 设计文档。
目标：找出设计漏洞 / 反例 / 实现陷阱 / 可证伪假设，不做客气评价。

文档位置：docs/m0-w2-p0-1-daemon-v03-design.md (相对仓根)
背景：LnMem PoC 14 天已完成，自学习闭环工程链已实现但 mismatch 反馈结构性短缺导致 Phase 2
门槛不可达。本设计为解此痛点，新增用户主动触发 mismatch 通道。

审查维度（必查）：
1. 设计目标是否真的解决问题陈述？是否存在隐含假设？
2. user_triggered 反馈的 frame_id 与 daemon 最近一帧对齐策略是否健壮？
3. trigger 阈值调整（brow_burst 0.85 → 0.7 + window 10 → 15）是否会过敏？历史
   PoC 数据 7 条 brow_burst 都是 brow≥0.9 全 match，放宽后会引入多少噪音？
4. user_triggered 反馈被加入训练批的 reward 默认 +1，是否会让模型过拟合到用户主观
   判断（用户也可能记错）？
5. feedback_to_batch.py 对 signal=null 反馈的处理路径是否会破坏现有 (sid, fid) 复合键
   slice_window 逻辑？
6. /health trigger_types metric 新加字段是否影响现有 client（如 watchdog）解析？
7. CLI 客户端 lnmem-tell 缺权限校验（任何能 curl 127.0.0.1:9876 的进程都可写入反馈），
   是否风险？
8. LNN-FIRST P-1 合规：新 trigger 类型有无破坏 LTC ODE forward 路径假设？
9. 测试方案覆盖率：是否有边界 case 漏掉？
10. 时间盒估算 4-6h 实现是否乐观？

输出格式：
- 每个 finding 标 CRITICAL / HIGH / MEDIUM / LOW
- 给出具体反例 / 代码片段证据 / 修复建议
- 最后一段 ## Summary 给 PASS / PASS-WITH-CHANGES / BLOCK 决议
```

---

## K. 签字栏

- cc 设计完成 V1：2026-05-11 09:00 CST ✅
- codex 对抗审一审：2026-05-11 09:06 CST ✅ → **BLOCK**（9 finding：2C/5H/2M），报告 `/tmp/codex-v03-review-output.md`
- cc 修订 V2（响应 9 finding）：2026-05-11 09:35 CST ✅
- codex 二审（可选）：⏳
- 川哥签字：⏳
- 实现启动：⏳

## L. V2 修订摘要（一图看完响应 codex 9 findings）

| Finding | 级别 | cc V2 响应 |
|---------|------|-----------|
| F1 reward 语义自相矛盾 | CRITICAL | §C.1 + §G 明确「按 REWARD_MAP 走，mismatch=-1，禁止 override」 |
| F2 signal=null 路径不成立 | HIGH | §C.3.3 重写：signal 仅审计用，训练仍走 (sid, fid) 切窗口；新加 4 metric |
| F3 5s 对齐缺契约 | HIGH | §C.1 + §C.3.1 加 schema 字段 trigger_ref_frame_ts / client_action_ts / alignment_latency_s + 硬过滤 |
| F4 brow 阈值放宽缺证据 | HIGH | §C.2 + §G6 收回 Phase A 不调阈值，拆 Phase B 离线回放 |
| F5 26 条统计口径未闭合 | HIGH | §A.bis 加复算命令 + 漂移历史 disambiguate |
| F6 HTTP 无鉴权 | CRITICAL | §C.3.1 加 X-LnMem-Token Header + secret_token 文件（chmod 600）+ 401 / 429 + user_triggered rate-limit 1/min |
| F7 trigger_types 口径 | MEDIUM | §C.3.4 明确口径关系 + watchdog 兼容性说明 |
| F8 测试漏边界 | HIGH | §D.3 V2 18 case（V1 10 case + 8 新边界 case） |
| F9 时间盒乐观 | MEDIUM | §F V2 拆 Phase A/B，Phase A 重估 10-13h（V1 7-9h） |
