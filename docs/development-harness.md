---
version: v0.1
audience: cc / codex / gemini / 工程师 / 文档贡献者
purpose: LnMem 软件开发工程纪律——基于产品目标定义验收 + 交付标准，按标准严格校验
status: 试运行 v0.1（2026-05-04 起）
parent: docs/STYLE.md（文档体裁规范）+ docs/rules/SCHEMA.md（万能公式 v1）
---

# LnMem 软件开发 Harness（BRO-PDCA × 目标驱动验收）

> **核心命题（老板原话锚定）**：**所有的开发工作，必须是基于产品要实现的目标制定验收和交付标准，完成开发工作后要按照标准进行校验和验收。**
>
> 翻译成工程动作：每个任务进入 D（执行）之前，必须先把"产品目标 → 可量化的 AC（验收标准）+ DoD（交付清单）"书面化；C（验收）必须按 AC + DoD 逐项校验，不允许"基本完成"等美化话术。**没有 AC + DoD 的开发等于无目标的散步。**

---

## 0. 三秒钟告诉你怎么开始

1. **接到任务**：跑 §2 BRO-PDCA 七步
2. **O 阶段是死结**：必须输出 **AC（验收标准）+ DoD（交付清单）**，缺则不允许进 P 阶段
3. **C 阶段是关门**：逐项 check AC + DoD，不达标必须如实报告"未达成 + 缺哪几项"，禁止说"基本完成 / 看起来 OK / 应该没问题"
4. **A 阶段是沉淀**：失败教训 → memory；不可妥协决策 → ADR；常态进展 → task-log

---

## 1. 核心铁律（P0 必读）

| RULE-ID | 触发 | 动作 | 反模式 | Why |
|---------|------|------|--------|-----|
| **HARNESS-AC-FIRST** | 收到任何开发/修改任务 | 在 O 阶段先书面化 **AC + DoD**（落到 commit message / issue / handoff，不允许只在脑里） | "我先写代码 AC 后面再补" | AC 是任务"可验收性"的唯一来源；事后补的 AC 一定通过，失去验证价值 |
| **HARNESS-AC-PRODUCT-ANCHOR** | 写 AC 时 | 每条 AC 必须能溯源到产品目标（不可妥协 6 项 / milestone 三色门槛 / 5 大核心承诺其中之一） | AC 写"代码跑通了"——只描述执行不描述价值 | "做完"和"做对"是两件事，必须锚回产品目标，否则做了无用功 |
| **HARNESS-AC-QUANTIFIED** | 写 AC 时 | 每条 AC 必须可量化：①数字 / ②命令 / ③产物 三选一 | "运行良好" / "性能 OK" / "看起来正确" / "基本达成" | 模糊 AC = 没 AC，cc 和老板对"达标"理解可能完全不一致 |
| **HARNESS-DoD-CHECKLIST** | 写 DoD 时 | 至少含 **代码 / 文档 / 沉淀** 三类清单（详见 §4） | "写完代码就交付" / "测试明天补" | 未沉淀的工作下次还会重新踩坑；未同步的文档会成为陷阱 |
| **HARNESS-CHECK-NO-SKIP** | 任务自认为完成时 | C 阶段必须逐项 check O 定义的 AC + DoD，输出"已达成 ✅ / 未达成 ❌ / 部分达成 ⚠️"清单 | C 阶段一句话"完成了" / 跳项不报 / 把"未达成"美化成"基本达成" | 跳项 = 老板被骗 = 最严重的工程红线（参考父仓 no-guessing.md 红线规则） |
| **HARNESS-CHECK-EVIDENCE** | C 阶段标注"已达成"时 | 必须附实测证据（命令输出 / 数字 / 截图 / 文件 diff），不允许"我看了下应该 OK" | "AC1 ✅"但没贴任何证据 | "幸存者偏差"——单次猜测式验证不算证据，必须可复现 |

**Why（这套铁律的根因）**：

1. **AI coding 的常见失败模式不是"实现错"而是"目标 drift"。** 任务跑半天发现"实现的不是老板要的"，根因是 O 阶段没把目标量化锚定到产品。
2. **老板验收时不可能逐行读代码**，只能根据 cc 自报的 C 结论判断。如果 C 不严格、cc 用"基本完成"美化，老板对项目状态的认知会失真，下游决策会错。
3. **沉淀（A）是单人 + AI 模式的唯一记忆通道。** 不沉淀 = 下次踩同坑（mycc 已踩过多次，参考各个 feedback memory）。

---

## 2. BRO-PDCA × LnMem 七阶段映射

> 整体来自父仓 mycc `3-Thinking/01-BRO-PDCA方法论.md` + LnMem 工程化加强（O 阶段强制 AC+DoD / C 阶段强制证据）。

### B (Briefing) — 任务接收

**触发**：从老板 / [`roadmap/milestones.md`](../roadmap/milestones.md) / 审查 finding / handoff 文档来的任务

**动作**：
1. **复述任务来源**：哪个 milestone？哪条 finding？老板原话？
2. **判断档位**（参考 mycc CLAUDE.md §2）：
   - **简洁档**（70%，<30min）：直接执行，但 O 阶段仍需 1-2 行 AC
   - **普通档**（20%，30-60min）：复述理解 + 简短确认 O / AC / DoD
   - **完整档**（10%，新系统/架构变更）：先输出 P1 拆解 + 完整 AC + DoD
3. **任务来源不清晰** → 先停下问老板，**禁止"猜测意图"开干**

**反模式**：模糊接到任务直接开工 → 实现完发现没用 → 浪费 2× 时间

### R (Reading) — 现状理解

**必读清单（按任务类型）**：

| 任务类型 | 必读 |
|---------|------|
| **改 contracts/** | [`docs/STYLE.md`](STYLE.md) + [`contracts/README.md`](../contracts/README.md) + 涉及的 SlotMeta / ProtocolDef |
| **改模块代码** | 该模块 README + 相关 contracts + 已有 ADR |
| **改 LNN 训练** | [`docs/lnn-math.md`](lnn-math.md) + [`docs/decisions/002-*.md`](decisions/) |
| **写文档** | [`docs/STYLE.md`](STYLE.md)（决定 L1-L4 体裁） |
| **任何任务** | [`CLAUDE.md`](../CLAUDE.md) §0 + [`roadmap/milestones.md`](../roadmap/milestones.md) 当前段 + [`docs/product-definition.md`](product-definition.md) §10.1（不可妥协 6 项） |

**反模式**：不读 STYLE.md 直接写文档 → 文档分叉 / 不读产品定义 §10.1 直接做妥协建议 → 红线违规

### O (Objective) — 目标对齐 ★ 死结环节 ★

**O1 输出**（What）：一句话祈使句，"做什么"

**O2 输出**（Why + Constraints）：为什么做（产品目标锚点）+ 不可妥协约束（如 §10.1 6 项 / milestone 三色门槛 / STYLE 文档体裁）

**AC 输出**（验收标准，铁律 HARNESS-AC-* 三条联合约束）：

> 每条 AC 必须满足：
> - **可溯源到产品目标**（指向不可妥协 6 项 / 5 大核心承诺 / milestone 红区门槛之一）
> - **可量化**（数字 / 命令 / 产物 三选一）
>
> 示例：
> - ✅ AC1: `pytest contracts/tests/ -v` 100% PASS（≥ 12 个测试）— 产品锚点：不可妥协项 §10.1.1 万能语法 schema 一致性
> - ✅ AC2: LNN τ 偏置训练 P95 收敛 epoch ≤ 50（数字）— 产品锚点：milestone M1 末"训练完全发散→红区"门槛
> - ✅ AC3: `docs/X.md` 新增 §N 含 **Why** 行（产物）— 产品锚点：STYLE.md L4 章节末 Why 强制规范
> - ❌ AC1: "代码运行良好"（无量化 + 无产品锚点）
> - ❌ AC2: "看起来 OK"（无量化）
> - ❌ AC3: "改好了"（无量化 + 无产品锚点）

**DoD 输出**（交付清单，铁律 HARNESS-DoD-CHECKLIST）：见 §4 模板

**反模式**：
- ❌ 跳过 O 阶段直接到 D
- ❌ 把 O 写成"我打算做 X"——这是 Plan 不是 Objective
- ❌ AC 不锚回产品目标——做完了没价值
- ❌ AC 模糊—— "看起来 OK" / "运行流畅" / "性能不错"

**O 阶段不通过 → 不能进 P 阶段。** 这是死结。

### P (Plan) — 工程拆解

**动作**：
1. 用 [`docs/STYLE.md`](STYLE.md) §3 L2 万能公式给任务打 **RULE-ID**（如 `M0-W3-LNN-TAU` / `CONTRACT-V1`）
2. 打 4 维评分（是否常用 / 是否必选 / P 级 / 权重）—— 见 [`docs/rules/SCHEMA.md`](rules/SCHEMA.md)
3. 决定执行模式：cc 主线 / codex 并行 / gemini 调研
4. 安排三道门（cc 实现 → codex 对抗审 → 人类签名）—— 见 [`CLAUDE.md`](../CLAUDE.md) §3

**反模式**：跳过 P 直接 D → 失去 RULE-ID 索引 → 后续召回失败

### D (Do) — 执行

**动作**：
1. **cc 主线**：写 contracts / 写骨架接口 / 关键决策
2. **codex 并行**（如适用，按 [`AGENTS.md`](../AGENTS.md) 边界）：填充实现
3. **gemini 调研**（如适用）：审查 / 文献综述，**不写代码**（[`GEMINI.md`](../GEMINI.md)）
4. **每一步小提交**：commit message 含 RULE-ID + "完成 AC 第 N 条"

**反模式**：D 阶段一气呵成几百行 → 失败时无法回滚到中间稳定点

### C (Check) — 严格验收 ★ 关门环节 ★

**动作**（铁律 HARNESS-CHECK-NO-SKIP + HARNESS-CHECK-EVIDENCE）：
1. 列出 O 阶段定义的所有 AC（编号）
2. **逐条 run 验证命令**——禁止"我看了下应该 OK"
3. 列出 DoD 清单（编号）
4. **逐条标注**：✅ 已达成 / ❌ 未达成 / ⚠️ 部分达成
5. **每个 ✅ 必须附证据**（命令输出 / 数字 / 文件 diff / 截图）
6. **未达成清单非空 → 必须如实在 commit message / 交付报告里写出来**

**输出格式**（commit message / 交付报告）：

```
## C 阶段验收报告

### AC 验收
- [x] AC1: contract test 100% PASS
      证据：`pytest contracts/tests/ -v` → 12/12 PASS
- [x] AC2: mypy --strict 0 error
      证据：`mypy contracts/ --strict` → Success: no issues found in 8 source files
- [ ] AC3: 文档同步 — ❌ 未达成
      证据：`grep -n 'M0-W3-LNN-TAU' contracts/README.md` → no match
      原因：忘记同步 contracts/README.md

### DoD 验收
- [x] 代码：tests + types + lint 全过
- [x] 文档：lnn-math.md 已加 §4.1 注解（diff 见 commit）
- [ ] 沉淀：task-log 已追加 / memory 未追加（无新教训）

### 结论
⚠️ 部分达成：AC3 文档未同步。建议：先补 contracts/README.md 再交付。
```

**反模式**：
- ❌ C 阶段一句话"完成了"
- ❌ 跳过未达成项不报
- ❌ 把"未达成"美化成"基本达成 / 大致 OK / 主要功能完成"
- ❌ 标 ✅ 但没贴证据
- ❌ 复用旧的（陈旧）测试输出当本次证据

### A (Adapt) — 沉淀

**沉淀位置**：

| 位置 | 触发 |
|------|------|
| `0-System/task-log.md`（父仓 mycc） | 任何完整档+ 任务（mycc 维度追踪） |
| `0-System/RECENT_EVENTS.md`（父仓 mycc） | 重要进展段（mycc 维度追踪） |
| `~/.claude/projects/-Users-rayman-chen-mycc/memory/feedback_*.md`（父仓 mycc） | 新教训（用户纠正 / 失败模式 / 反直觉成功） |
| `LnMem/docs/decisions/*.md` | LnMem 不可妥协决策（含 ADR 模板） |

**反模式**：完成任务直接关 session 不沉淀 → 下次踩同坑（LnMem M0 W2 已多次踩坑）

---

## 3. AC 模板（按任务类型）

### 3.1 代码任务 AC 模板

```
AC1: 单元测试 — `pytest tests/<module>/ -v` 100% PASS（X 个测试）
     产品锚点：<指向不可妥协项 / milestone 门槛 / 核心承诺>
AC2: 类型检查 — `mypy <module>/ --strict` 0 error
     产品锚点：契约一致性（不可妥协项 §10.1.1）
AC3: Lint — `ruff check <module>/` 0 violation
AC4: 集成无 break — grep 全消费方无 type/signature 冲突
AC5: 性能（如适用）— 关键路径 P95 < <数字>ms
     产品锚点：milestone 三色门槛中"<具体场景>性能门槛"
```

### 3.2 文档任务 AC 模板

```
AC1: L1 frontmatter 含 version + audience + purpose 三必填
     产品锚点：STYLE.md §2 L1 共用骨架
AC2: 决策点和反直觉点章节末有 **Why**:
     产品锚点：STYLE.md L2/L4 章节末 Why 强制规范
AC3: 链接全部本地相对路径或完整 URL（grep 无死链）
     产品锚点：父仓 feedback_pr_clean_branch_isolation 实战教训（vendored 跨仓死链问题）
AC4: 体裁分级符合 STYLE.md（L2 规则集 / L4 叙述类）
AC5: cc 抽样自评通过（跨章节论证一致性）
```

### 3.3 架构/契约决策 AC 模板

```
AC1: ADR 已落档（docs/decisions/<NNN>-<title>.md）
     产品锚点：CLAUDE.md §3.1 cc/codex 冲突诊断 ADR 流程
AC2: 派生规则有 RULE-ID（≥ 1 条 P0/P1）
AC3: contracts/ 接口已锁（如适用）
     产品锚点：不可妥协项 §10.1.1 万能语法 schema 一致性
AC4: 三道门已过（cc + codex 对抗审 + 老板签名）
     产品锚点：CLAUDE.md §3 三道门
AC5: 不可妥协 6 项无回退（grep product-definition.md §10.1 确认）
     产品锚点：红线规则
```

---

## 4. DoD 模板（按任务类型）

### 4.1 代码任务 DoD

```
代码层：
- [ ] 所有新增/修改函数有单元测试
- [ ] 测试覆盖率 ≥ 80%（关键路径 ≥ 95%）
- [ ] mypy --strict 通过
- [ ] ruff check 通过
- [ ] 所有 contract 边界已遵循 contracts/

文档层：
- [ ] 模块 README 同步更新（如改了 API）
- [ ] CLAUDE.md / AGENTS.md 同步（如改了 AI 调用约定）
- [ ] CHANGELOG（如有大改动）

沉淀层：
- [ ] mycc task-log 已追加
- [ ] mycc memory（如有新教训）
- [ ] LnMem ADR（如有不可妥协决策）

集成层：
- [ ] 三道门：cc 实现 + codex 对抗审 + 老板签名
- [ ] 公网/远端服务（如适用）：实际部署 + curl 验证
```

### 4.2 文档任务 DoD

```
文档层：
- [ ] L1 frontmatter 三必填字段
- [ ] 决策点 / 反直觉点章节末有 Why
- [ ] 链接全本地相对路径或完整 URL
- [ ] 体裁分级符合 STYLE.md

引用层：
- [ ] 5 工具入口（CLAUDE/AGENTS/GEMINI/.cursorrules/.github）已加索引（如新建 P0/P1 文档）
- [ ] README "30 秒入口"已更新（如新建主线文档）

沉淀层：
- [ ] mycc task-log 追加
- [ ] mycc memory（如学到新写作规律）
```

### 4.3 架构/契约决策 DoD

```
决策层：
- [ ] ADR 落档（docs/decisions/<NNN>-<title>.md）
- [ ] ADR 含 5 findings + 综合评级（GREEN/YELLOW/RED）
- [ ] 派生规则有 RULE-ID（如 P0/P1）

接口层：
- [ ] contracts/ 边界冻结（如适用）
- [ ] 调用方无 break（grep 全消费方）

文档层：
- [ ] product-definition.md / source-research.md 同步（如影响产品定义）
- [ ] roadmap/milestones.md 同步（如影响节奏）

沉淀层：
- [ ] mycc task-log 追加
- [ ] mycc memory（如反直觉决策点）
```

---

## 5. 任务模板（commit message / issue / handoff 用）

```
<type>(<scope>): <title>

## O (Objective) ─ 任务目标
- O1（What）: <一句话做什么>
- O2-Why: <为什么做，锚回产品目标>
- O2-Constraints: <不可妥协约束>

## AC（验收标准，HARNESS-AC-* 三铁律）
- AC1: <可量化条件 + 产品锚点>
- AC2: ...

## DoD（交付清单，HARNESS-DoD-CHECKLIST）
- [x] 代码：...
- [x] 文档：...
- [x] 沉淀：...

## C (Check) ─ 验收报告（HARNESS-CHECK-NO-SKIP + EVIDENCE）
- ✅ AC1 已达成（实测 ...，证据：...）
- ✅ AC2 已达成（实测 ...）
- ⚠️ AC3 部分达成（缺 ...）

## A (Adapt) ─ 沉淀
- task-log: <文件路径>
- memory: <如有新教训，路径>
- ADR: <如有不可妥协决策，路径>
```

---

## 6. 反模式（禁止）

| 反模式 | 危害 | 已知踩坑 |
|--------|------|---------|
| 跳过 O 直接 D | 目标 drift / 实现完发现没用 | mycc 历史多次（参考 EvoDesk session） |
| AC 不锚产品目标 | 做完了没产品价值 | EvoDesk V1.6→V1.15 后端狂奔但前端 0 增量（feedback_product_outcome_anti_drift） |
| AC 模糊（"运行良好"） | C 阶段无法严格验收 | mycc CLAUDE.md 早期 |
| C 阶段美化"未达成" → "基本达成" | 老板对状态认知失真 → 后续决策错 | 红线（父仓 no-guessing.md） |
| 标 ✅ 但无证据 | 幸存者偏差 / 假性完成 | mycc EvoDesk B17.x 多次 |
| 三道门跳过 codex 对抗审 | 单视角偏差 / 漏洞流入 | LnMem M0 W2 codex 配额满险些跳过（feedback_review_engine_3level_fallback） |
| A 阶段不沉淀 | 下次踩同坑 | LnMem M0 W2 多次 4 路 codex 经验未即时落档 |
| commit message 不含 O / AC / DoD / C 四段 | 老板看不懂任务边界 | mycc 历史 |

---

## 7. 试运行检查表（M0 末复盘）

本 harness 启动一周后（≈ 2026-05-11）检查：

- [ ] 所有 commit message 含 **O / AC / DoD / C** 四段（合规率 ≥ 90%）
- [ ] AC 可量化率 ≥ 95%（≤ 5% 模糊 AC 是可接受灰度）
- [ ] AC 锚回产品目标率 ≥ 85%（指向不可妥协项 / milestone 门槛 / 核心承诺）
- [ ] C 阶段未达成项被如实报告率 = **100%**（任何美化都是红线违规）
- [ ] C 阶段 ✅ 项附证据率 = 100%
- [ ] mycc task-log 追加率 ≥ 90%
- [ ] 老板抽查"任务边界 vs 实际产出"一致性 ≥ 85%

**任一项不达标** → root cause 分析 + 调整 v0.1 → v0.2
**全过** → 升级 status 从"试运行"到"稳定"
**回归严重** → 回退到 v0.1 之前（即不强制本规范）

---

## 8. 与 mycc CLAUDE.md / 其他规范的关系

| 维度 | 来源 | 在本 harness 中的作用 |
|------|------|------|
| BRO-PDCA 七阶段方法论 | mycc `3-Thinking/01-BRO-PDCA方法论.md` | 直接复用，本 harness 把 O 阶段强制 AC+DoD、C 阶段强制证据 |
| 三道门 | LnMem [`CLAUDE.md`](../CLAUDE.md) §3 | DoD"集成层"复用，作为强制项 |
| 万能公式 v1（4 维 / RULE-ID） | LnMem [`docs/rules/SCHEMA.md`](rules/SCHEMA.md)（vendored mycc） | P 阶段给任务打 RULE-ID 用 |
| L1-L4 体裁分级 | LnMem [`docs/STYLE.md`](STYLE.md) | 文档任务 AC + DoD 复用 |
| 不可妥协 6 项 | LnMem [`docs/product-definition.md`](product-definition.md) §10.1 | AC 产品锚点 + 反模式红线 |
| no-guessing 红线 | mycc `~/.claude/rules/no-guessing.md` | C 阶段"未达成必须如实报告"红线来源 |

**Why 不直接 import mycc CLAUDE.md**：mycc 是 cc 全局工作配置（覆盖所有任务包含运维/调试/沟通），本 harness 是 LnMem 项目级软件开发纪律——**对"目标可验证性"的工程纪律要求更严**（强制 AC+DoD 书面化 + 证据），需要单独成文。

---

## 9. 本 harness 的诞生本身就是对 harness 的示范

> 老板要求："给这个项目加上软件开发的约束 harness ，严格按照 BRO-PDCA 工作方法论..."
>
> 本文档的开发过程严格遵循自身定义的 BRO-PDCA：
> - **B**: 老板原话"加上软件开发的约束 harness"
> - **R**: 读 mycc BRO-PDCA 方法论 + LnMem CLAUDE.md / STYLE.md / SCHEMA.md
> - **O**: 给 LnMem 加文档级 harness，AC = "5 工具入口都加索引 + 主文档可读 + 老板抽查认同"
> - **P**: L2 万能公式格式 / 9 段结构 / 配套 5 入口修改
> - **D**: 写 development-harness.md + 改 5 入口 + 改 README
> - **C**: 远端拉回验证 + 链接可达 + grep 入口索引齐全
> - **A**: task-log 追加 + memory 加 feedback_lnmem_dev_harness

---

**v0.1 起草：cc | 2026-05-04 | 等老板拍板试运行 v0.1 → 2026-05-11 复盘升 v0.2 或回退**
