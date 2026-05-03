---
version: v0.1
audience: cc / codex / 工程师 / 文档贡献者
purpose: LnMem 项目所有 .md 文件的统一写作规范——4 级体裁分层 + L1 frontmatter + L2 万能公式 v1
status: 试运行 v0.1（2026-05-04 起）
parent: mycc/docs/rules/SCHEMA.md（万能公式 v1 源规范）
---

# LnMem 文档写作规范（STYLE）

> **核心命题**：用万能公式 v1 做"持续工程化文档生产"，但**分级而非一刀切**——LnMem 50% 是叙述/教学/研究文档（产品定义 / 数学手册 / 研究底稿），强行套 RULE-ID 反而损害可读性。
>
> 本规范在 [mycc/docs/rules/SCHEMA.md](https://github.com/Hahaxiong838383/mycc) 的 4 维矩阵 + RULE-ID 命名规范之上，按 LnMem 文档体裁加一层 L1-L4 分级。

---

## 1. 4 级体裁分层（核心决策表）

| 级别 | 适用文档 | 强制规则 | 行数控制 | LnMem 实例 |
|------|---------|---------|---------|----------|
| **L1 共用骨架** | 全部 .md | YAML frontmatter（version / audience / purpose）+ 章节末 Why（重要章节） | — | 所有 19 个 .md |
| **L2 万能公式 v1 完整** | 规则 / 里程碑 / 模块 README | L1 + RULE-ID + 4 维分 + P0-P3 + 索引下沉 | ≤ mycc SCHEMA.md 标准 | CLAUDE.md / AGENTS.md / contracts/README.md / roadmap/milestones.md / 4 模块 README / spike README |
| **L3 半结构化** | ADR / 入门 / 导航 | L1 + 头部 RULE-ID 索引表（如有规则）+ 主体保留现模板 | 不强制 | docs/decisions/*.md / docs/onboarding.md / docs/glossary.md |
| **L4 自由叙述 + 元数据** | 产品定义 / 研究 / 技术 / 数学教学 | L1 + 章节末 Why + 关键决策表；**不强制 RULE-ID** | 不强制 | product-definition.md / source-research.md / memory-architecture.md / lnn-math.md / README.md |

**Why 这套分级**：cc/codex 高频读写的"规则集"享受密度优化；投资人/新人看的"门面材料"保住可读性；ADR/导航折中（结构 + 自由）。一刀切两边都伤。

---

## 2. L1 共用骨架（所有 .md 必填）

### 2.1 YAML frontmatter 模板

```yaml
---
version: v0.X.Y         # 文档版本号（独立于项目版本，但可对齐）
audience: <谁会读>       # cc / codex / 工程师 / 投资人 / 新人 / 团队 / 历史读者
purpose: <一句话定位>    # 这个文档解决什么问题，不超过 20 字
status: <稳定 | 试运行 | deprecated | 草稿>
last_updated: 2026-05-XX # 仅大改时更新（小修不动）
---
```

**最少字段**：`version` + `audience` + `purpose`（其他可选）。

### 2.2 章节末 Why（L2/L3/L4 重要章节都要写）

格式：在章节最后一段或表格下方，加一句 `**Why**：<原因>`。

**反模式**：
- ❌ 把 Why 砍成"为了清晰"这种空话
- ❌ 每个二级标题都写 Why（信号噪音化）—— 只在"决策点"和"反直觉点"写

**Why（这条规则的 Why）**：让边缘情况可判断。Why 是"判断未来类似情况"的钥匙，不是修辞。

---

## 3. L2 万能公式 v1（规则集专用）

完整规范见 [mycc/docs/rules/SCHEMA.md](https://github.com/Hahaxiong838383/mycc)。LnMem 直接复用：

- **4 维评分**：是否常用 / 是否必选 / P 级（P0-P3）/ 权重（0.35-1.00）
- **P 分级**：P0 = 全文必读 / P1 = 速览表 / P2/P3 = 索引下沉
- **RULE-ID 命名**：大写 + 连字符，3-15 字符，全局唯一（如 `M0-W2-CONTRACT` / `CODEX-PARALLEL`）
- **P0/P1/P2/P3 标准格式**：见 SCHEMA.md §"单条规则的标准格式"

**LnMem 特有的 RULE-ID 领域前缀**：
- `M0-` / `M1-` / ...：里程碑相关规则
- `CONTRACT-`：contracts/ 边界规则
- `CODEX-`：4 路 codex 协作纪律
- `EVAL-`：评估/验收规则
- `LNN-`：LNN 训练相关约束

---

## 4. L3 半结构化（ADR / 入门 / 导航）

### 4.1 ADR（决策记录）

保留现模板（[`docs/decisions/000-template.md`](decisions/000-template.md)），**不引入 RULE-ID 到正文**，但在文档头部加 L1 frontmatter 后**附一个简短 RULE-ID 表**（如有规则化产物）：

```yaml
---
version: v1
audience: 历史读者 / 未来 cc/codex
purpose: <ADR 标题简版>
status: 已记录 / 已废弃 / 已修订
---

# ADR NNN — <标题>

| 派生规则 RULE-ID | P 级 | 触发 | 文件 |
|-----------------|------|------|------|
| LNN-TAU-POSITIVITY | P0 | LNN 训练含 τ 偏置 | docs/rules/LNN-TAU-POSITIVITY.md |

（如本 ADR 没有派生规则，省略此表）

## 背景 / 5 个 findings / 综合评级 / 后续行动
（保留现 ADR 写法）
```

### 4.2 入门 / 导航（onboarding.md / glossary.md）

L1 frontmatter + 章节链接列表为主，**不强制 RULE-ID**。术语表（glossary）密度天然高，按"术语 — 一句话定义 — 出处链接"三栏即可。

---

## 5. L4 自由叙述 + 元数据（产品/研究/技术/数学）

**强制**：L1 frontmatter + 章节末 Why（决策点和反直觉点）+ 顶部目录（≥ 200 行的文档）。

**不强制**：RULE-ID / P 分级 / 4 维评分。这些文档的**目标受众是人类阅读连贯性**，硬套 RULE-ID 会摧毁论证链。

**示范（已在用）**：
- `product-definition.md` v0.3.2 每章末 `**Why**：` 已自带优先级语义
- `source-research.md` 每节末有"v 阶段标注 + 依据"
- `lnn-math.md` §0.1 有"通俗类比 + 比喻总结表"

---

## 6. 新增/修改文档的 4 步工作流

```
第 1 步：判断体裁 → 落到 L1/L2/L3/L4 哪一级（看本文 §1 表）
第 2 步：写 L1 frontmatter（≥ version/audience/purpose 三字段）
第 3 步：按对应级别的格式写正文
第 4 步：自检反模式（见 §7），过了再 commit
```

**修改既有文档**：
- 改正文 → frontmatter 的 `last_updated` 同步改（仅大改）
- 改 P 级 → 同步移动文件位置（P0↔P1 在主文件 ↔ docs/rules/）
- 改 RULE-ID → 全仓库 grep 替换（包括 README / 飞书引用）

---

## 7. 反模式（禁止）

| 反模式 | 危害 | 案例 |
|--------|------|------|
| 把 L4 文档（产品定义）硬套 L2 RULE-ID | 论证链断裂，投资人看不下去 | 反例：把 §1.3 五条产品承诺改成 RULE-ID 表 |
| frontmatter 的 `audience` 写"all" / "everyone" | 等于没写，无法决定密度优先级 | 改为具体角色组合：`cc / codex / 工程师` |
| 跨 session 对同一规则 P 级评分漂移 | 文档分叉，cc 召回失败 | 见 mycc `feedback_universal_formula_v1.md` |
| L2 规则的 Why 被砍掉 | 边缘情况靠拍脑袋 | mycc CLAUDE.md 已踩过坑 |
| 同一 RULE-ID 在主文件和子文件都有详情 | 一处更新另一处过期 | 见 mycc SCHEMA.md §反模式 |
| L4 文档写了 frontmatter 但章节末没 Why | 等于"有壳无核"，仍然是散文 | — |

---

## 8. 试运行检查表（M0 末复盘）

L2 启动一周后（≈ 2026-05-11）检查：

- [ ] L2 文档（CLAUDE.md / AGENTS.md / contracts/README.md / 4 模块 README）行数压缩比 ≥ 40%
- [ ] L1 frontmatter 覆盖全部 19 个 .md（含 L4 类）
- [ ] L4 文档可读性无回归（cc 抽样自评 + 老板反馈）
- [ ] cc 跨 session 对同一规则 P 级评分一致率 ≥ 85%（语料抽 5 条）
- [ ] 新增文档时 cc 实际遵循率 ≥ 80%（不绕过 L1 frontmatter）

**任一项不达标** → 在该项对应的环节做 root cause 分析，调整本规范 v0.1 → v0.2。
**4 项全过** → 升级 status 从"试运行"到"稳定"。
**回归严重** → 回退到 v0.1 之前的写法（即不强制本规范）。

---

## 9. 与 mycc 万能公式 v1 的关系

| 维度 | mycc SCHEMA.md | LnMem STYLE.md（本文档） |
|------|---------------|------------------------|
| 适用范围 | mycc 项目所有"行为规则" | LnMem 项目所有 .md 文档 |
| 4 维矩阵 | 定义者 | 直接复用 |
| RULE-ID 规范 | 定义者 | 直接复用 + 加 LnMem 领域前缀 |
| L1-L4 体裁分级 | 无（mycc 几乎全是规则） | **本规范新增**（适应 LnMem 50% 叙述类文档） |
| 反模式清单 | 5 条 | 复用 5 条 + 加 LnMem 特有 1 条（L4 强套 L2） |

**Why 不直接 import mycc SCHEMA.md**：mycc 是规则项目，LnMem 是产品+研究+工程混合体，文档体裁分布不同；本规范只在 mycc 之上加分层，避免重写。

---

**v0.1 起草：cc | 2026-05-04 | 等老板拍板试运行**
