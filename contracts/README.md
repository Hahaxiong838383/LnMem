# contracts/ · 公共数据契约

> 4 路 codex 跨模块工作的唯一共享 API 表面。模块之间只通过 `contracts.*` 交换数据 — 内部表示自由。

## 文件清单

| 文件 | 内容 |
|------|------|
| [`schema_v1.py`](schema_v1.py) | 万能语法 20 槽 schema_v1（M0 草案 / M1 末冻结） |
| [`event_tensor.py`](event_tensor.py) | `EventTensor` + `SlotValue` 数据类 + 不可变更新语义 |
| [`protocols.py`](protocols.py) | 4 模块的 Protocol 接口（duck typing） |
| [`__init__.py`](__init__.py) | 公共导出，所有别处只 `from contracts import ...` |

## 边界规则

1. **修改 contracts/ 必须经老板签字**（[`../CLAUDE.md`](../CLAUDE.md) §5 不可妥协 §10.1#1 万能语法）。
2. **新增字段保持向后兼容**：dataclass 加默认值；不允许改默认行为。
3. **不可变更新**：`EventTensor.set()` 返回新对象，禁止原地改 `slots`（[`~/.claude/rules/coding-style.md`](https://example/) 全局规则 immutability）。
4. **schema_v1 槽位数严格 = 20**：`schema_v1.py` 末尾有 assert，CI/import 时即触发。

## 为什么 schema 在 cc 主线写

vibecoding 三道门第 1 道：接口先行。如果 4 路 codex 各自定义 EventTensor，会出现字段漂移，上游下游对不上（参考父仓 `feedback_codex_multi_field_contract_gate`）。cc 主线先冻结 contracts，才并行开 codex。

## 后续

- M0 W2-W4：4 路 codex 在各自模块内实现 Protocol（占位 → 真跑通）
- M1 末：schema_v1 冻结为 v1；之后变更必须 v1.x patch + ADR
- M3 末：基于真实 evals 数据回看 schema_v1 是否需要 v2
