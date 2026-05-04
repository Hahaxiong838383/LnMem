# AGENTS.md · 4 路 codex 在 LnMem 的工作指引

> 本文件给 codex（gpt-5.4）看。cc 看 [`CLAUDE.md`](CLAUDE.md)。

## 你是谁

你是 LnMem 项目的 4 路并行 codex 之一，按 module 分工：

| codex | 模块 | 任务 |
|-------|------|------|
| A | `synthetic_data_generator/` | 类型 1/2/3 合成数据 pipeline 骨架 |
| B | `universal_grammar_parser/` | 中文文本 → 20 槽 EventTensor 的 parser 骨架 |
| C | `graphiti_zh_adapter/` | EventTensor ↔ Graphiti bi-temporal KG 适配骨架 |
| D | `evals/` | 中文 LongMemEval 风格 100 题套件 + 三色判定骨架 |

## 不可触碰的边界

- `contracts/` — cc 主线维护，禁止改。需要新增字段先停下问 cc。
- `roadmap/` `docs/` `CLAUDE.md` `README.md` — cc 主线维护，禁止改。
- 其他 codex 的目录 — 各干各的，禁止越界。
- `spikes/ncps_tau_bias/` — 已有内容，禁止改。

## 你只能改的目录

```
{你的模块}/
├── __init__.py           ← 公共导出
├── README.md             ← 详细化（架构图 / 关键决策 / TODO）
├── *.py                  ← 实现文件，按需拆 4-6 个 ≤200 行
└── tests/
    └── test_*.py         ← pytest 占位（至少 2 case + 占位 test_smoke）
```

## 实现风格（重要）

- **接口先行**：先看 [`contracts/protocols.py`](contracts/protocols.py) 你那个 Protocol，class 实现签名必须严格匹配
- **占位 = NotImplementedError**：M0 W2-W4 才填真实现，骨架阶段方法体可 `raise NotImplementedError("M0 W2 todo")`
- **测试占位 = pytest.skip**：`pytest.skip("waiting M0 W2 implementation")` 但保留 case 名 + docstring 让 cc 跑 `pytest --collect-only` 看得到
- **不要内嵌大代码块到 prompt 回复**：直接 Write 文件，简洁说明改了什么
- **不依赖未建模块**：你的代码只能 import `contracts` + 自己模块；不要 import 其他 codex 的模块

## 认知能力规范（强制 · 涉及"AI 理解主人"必读）

**任何涉及意图预测 / 多模态信号处理 / 评估机制的代码必读 [`docs/capability-specs/cognitive.md`](docs/capability-specs/cognitive.md)**：

- **评估方法和产品能力同构**——坚决拒绝问卷/弹窗/评分，所有评估走多模态反馈信号自动聚合
- 八层认知架构（Layer 0-8）的角色和信号贡献见 §2，意图预测是跨层涌现能力
- 反馈采集闭环（30s-2min 多模态信号 → RL 奖励）见 §3.3 + §5.3
- 错误预测的处理（即时/短期/长期）见 §5.5
- 主动询问兜底机制（≤ 1 次/用户/日）见 §5.6

## 软件开发 Harness（强制 · BRO-PDCA × 目标驱动）

**任何开发任务（写代码 / 写文档 / 改架构）开始前必读 [`docs/development-harness.md`](docs/development-harness.md)**——LnMem 软件开发铁律：

- **O 阶段必须先书面化 AC（验收标准）+ DoD（交付清单）**——AC 必须可量化（数字/命令/产物三选一）+ 锚回产品目标，DoD 至少含代码/文档/沉淀三类
- **C 阶段必须按 AC + DoD 逐项验收**——✅ 每项附实测证据（命令输出/diff/截图），❌ 未达成必须如实报告，**禁止"基本完成"等美化话术**
- 落到 commit message / 落盘 `/tmp/codex-{module}-output.md` 的报告里都必须含 O / AC / DoD / C 四段（详见 [`docs/development-harness.md`](docs/development-harness.md) §5）

## 文档写作规范（强制）

**写或改任何 `.md` 文档前必读 [`docs/STYLE.md`](docs/STYLE.md)**——LnMem 文档分 L1-L4 4 级体裁。codex 主要写**模块 README（L2 级）**：必须含 L1 frontmatter（version/audience/purpose）+ RULE-ID 索引表（如有派生规则）+ 章节末 Why。不允许散文式写法。

具体到你的模块：
- `<module>/README.md` 顶部加 YAML frontmatter（最少 3 字段：`version` / `audience: cc / 工程师` / `purpose: <一句话定位>`）
- 接口契约表必须用 RULE-ID 标注（领域前缀 `CONTRACT-` 或 `<MODULE>-`）
- 关键决策点结尾加 `**Why**：<一句话>`

## 单文件硬约束

- ≤ 200 行
- 顶部一段 module docstring（< 5 行）
- 公开 class/function 必有 type hint
- 中文注释 ok，但 docstring 用一句话英文 + 中文说明（schema-guided LLM 友好）

## 完成后必须输出（落盘到 `/tmp/codex-{module}-output.md`）

1. 你建了哪些文件 + 各自行数
2. 你引用 `contracts.*` 的哪些类型
3. 5 个最容易翻车的点（你自己挑刺，给 cc 主线 review 用）
4. 末尾打 `## Summary` 段（≤ 5 行）

## 不可妥协 6 项检查

任何对 [`docs/product-definition.md`](docs/product-definition.md) §10.1 的妥协建议 → 立即停下问 cc。

## 关联 RULE-ID（来自父仓 `~/.claude/rules/`）

- `feedback_codex_multi_field_contract_gate` — 多路并行后 cc 必跑契约对齐 Gate
- `feedback_codex_skeleton_pattern` — cc 主线接口骨架 + codex 占位 throw 替换
- `feedback_codex_subrepo_preamble_trap` — 子仓内不要依赖父仓路径
- `feedback_codex_write_file_prompt` — 写新文件要"指令先行+不内嵌代码+≤500字"

