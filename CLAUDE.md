# CLAUDE.md · LnMem 仓库的 vibecoding SOP

> **这份文档是给 Claude Code 和 Codex 看的工作纪律**——人类工程师也读，但每条规则的执行主体是 AI 同事。
> 编制：cc + 老板 | 2026-05-03 | 配套 v0.3 封版产品定义

---

## 0. 三秒钟告诉你怎么开始

1. 第一次到这个仓库：先读 [`README.md`](README.md) 的"30 秒入口"段
2. **写或改任何 `.md` 文档前**：必读 [`docs/STYLE.md`](docs/STYLE.md)（4 级体裁分层 + L1 frontmatter + L2 万能公式 v1）—— **这是强制规范，不是建议**
3. 准备开始一个工程任务：先读 [`roadmap/milestones.md`](roadmap/milestones.md) 当前月份段
4. 准备写代码：执行下面 §3 的"三道门"
5. 任何对不可妥协 6 项的妥协建议：**先停下问老板**

---

## 1. AI 同事的责任边界

| 主体 | 责任 |
|------|------|
| **Claude Code（cc）** | 产出代码 / 文档 / 测试用例。**不是责任主体** |
| **Codex（gpt-5.4）** | 对抗审查 / 边界条件提示 / 工程规范深挖。**不是责任主体** |
| **对应人类工程师** | 所有 AI 产出的 PR、文档、测试，必须由人类工程师署名确认后才合并 |
| **架构 lead（老板）** | 跨模块决策的最终责任人 |

**核心原则**：AI 是工具不是替罪羊。这条不是法律免责，是工程纪律——避免出 bug 时大家互相推给 AI。

---

## 2. vibecoding 工作流

### 2.1 PR 三道门（每份代码必经）

1. **cc 实现** —— 写代码、写测试、写文档
2. **codex 对抗审查** —— 找最容易翻车的 5 个点
3. **人类工程师签名** —— 实质审查后合并

不允许跳过任何一道。

### 2.2 文档双引擎

- cc 主写
- codex 提对立观点
- 人定调

### 2.3 Milestone 启动前置防御

每个月 milestone 启动前，codex 出"5 个最容易翻车的点"清单——cc 把它们写进里程碑测试用例。

### 2.4 周复盘

每周一次 cc + codex 联合 retro：上周 PR 质量、协作摩擦、调整建议。

---

## 3. 任何模块开始前的"三道门"

写新模块前必须按这个顺序：

1. **接口先行**：先写 Protocol（Python `typing.Protocol` 或 TypeScript `interface`）+ 输入输出 schema，再写实现
2. **测试先行**：单元测试定义验收标准，TDD 节奏（红→绿→重构）
3. **codex 对抗审查**：cc 写完立即让 codex 挑刺，再 commit

**反模式**（禁止）：
- 直接写实现不写 Protocol
- 测试事后补
- cc 自己 review 自己

### 3.1 cc / codex 意见冲突诊断流程（v1.1 新增）

> 真正的风险不是"AI 写错代码"，是"老板懒得读 AI 的对抗意见"。

当 cc 主线 和 codex 对抗审查给出**相反建议**时：

1. **默认采纳 codex 的对抗审查意见**（保守优先）
2. **cc 不同意必须明确分歧点**——不许"我觉得 ok"，必须列出：codex 担心什么 / cc 凭什么不同意 / 实测证据是什么
3. **老板作为最终决策者**，但**必须留下决策记录**到 [`docs/decisions/`](docs/decisions/)（ADR 风格，每个冲突一份 markdown）
4. **每周 retro 回看**所有冲突案例：3 个月后能复盘"我们和 codex 谁更准"，迭代协作 SOP

ADR 文件命名：`docs/decisions/NNN-brief-title.md`（NNN 三位数顺序号）。模板见 [`docs/decisions/000-template.md`](docs/decisions/000-template.md)。

---

## 4. PR 描述模板

每个 PR 描述必须含：

```markdown
## 改动摘要
[1-2 句话]

## AI 协作贡献
- cc：[做了什么]
- codex：[审查结果 / 提的关键点]

## 人类工程师确认人
@<github-username>

## 测试
- [ ] 单元测试通过
- [ ] codex 对抗审查通过
- [ ] 关联里程碑：M? §?

## 不可妥协项检查
- [ ] 无对 v0.3 §10.1 六项的妥协
- [ ] 无对 §13.1 双门验证标准的修改
```

---

## 5. 何时应该停下来问老板

**红线（强制停下）**：

- 任何对 [`docs/product-definition.md`](docs/product-definition.md) §10.1（不可妥协 6 项）的妥协建议
- 任何对 §13.1 双门验证标准的修改建议
- 三色判定（[`roadmap/milestones.md`](roadmap/milestones.md)）连续两次进入红区
- 类型 3 数据使用边界的灰色地带
- 用户访谈反馈与 §13.1 双门验证标准冲突
- 老板未在当前对话内授权过的"对外操作"（推送公开 GitHub / 发飞书 / 给用户发消息等）

**黄区（建议停下问）**：

- 单 PR 改动跨 3 个以上模块
- 单 PR 测试覆盖率 < 70%
- codex 审查给出 "high" 级别 finding
- 性能指标偏离设计目标 > 20%

---

## 6. 调用 Codex 的标准 prompt 模板

cc 调 codex 时必须用这个模板（精简版，详见父仓 `feedback_codex_prompt_template`）：

```
# 任务：<一句话目标>

## 上下文（≤200 字）
项目：LnMem · LNN-centric memory system
当前模块：<module-name>
关联 RULE-ID / 里程碑：<§ref>

## 输入
<具体代码 / 接口 / 测试 case，含完整路径>

## 期望输出（落盘必须）
- 输出位置：/tmp/codex-<task>-output.md
- 末尾必须打 ## Summary 段（不超 5 行）

## 判断标准
- 必须给出对抗性审查（5 个最容易翻车的点）
- 不要重写代码，只指出问题 + 建议方向
- 引用 LnMem 文档时用相对路径
```

---

## 7. 文档命名 / 修改约定

| 类型 | 位置 | 命名 |
|------|------|------|
| 产品定义级文档 | `docs/` | 全小写连字符 |
| 技术架构文档 | `docs/` | 全小写连字符 |
| 路线图 / 里程碑 | `roadmap/` | 全小写连字符 |
| AI 协作产出 | 落盘到 `/tmp/` | 临时文件，不进 git |

**修改产品定义 v0.3** 必须经过老板签字——这是封版文档。任何修订都升 v0.x.1 并写变更摘要到 §14。

---

## 8. 链接跳转

- [`README.md`](README.md) - 项目入口
- [`docs/product-definition.md`](docs/product-definition.md) - v0.3 封版产品定义
- [`docs/memory-architecture.md`](docs/memory-architecture.md) - 技术架构
- [`docs/source-research.md`](docs/source-research.md) - 研究底稿
- [`docs/glossary.md`](docs/glossary.md) - 术语字典
- [`roadmap/milestones.md`](roadmap/milestones.md) - M0-M12 路线图

---

**v0.3 封版同步 · 2026-05-03**
