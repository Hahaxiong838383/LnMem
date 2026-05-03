# LnMem · 30 分钟新成员上手手册

> 给将来加入项目的人（或 cc/codex 接手新 session）用。**M3 末扩团队前必须完整。**
>
> 当前版本：v0.1（M0 W1 起步骨架，M0 末完整化）

---

## 你只有 30 分钟？读这个最小路径

1. **2 分钟看产品愿景**：[`README.md`](../README.md) §它是什么 + §核心承诺
2. **5 分钟看技术架构**：[`README.md`](../README.md) §核心架构（一图）+ 分层详解前 3 层
3. **5 分钟看你这周该做什么**：[`roadmap/milestones.md`](../roadmap/milestones.md) 当前月份段
4. **3 分钟看协作纪律**：[`CLAUDE.md`](../CLAUDE.md) §3 三道门 + §3.1 cc/codex 冲突诊断
5. **3 分钟看不可妥协 6 项**：[`docs/product-definition.md`](product-definition.md) §10.1
6. **2 分钟看红线**：[`README.md`](../README.md) §给 AI 同事的 30 秒入口
7. **10 分钟读历史决策**：[`docs/decisions/`](decisions/) 按编号顺序读完所有 ADR

---

## 项目结构地图

```
LnMem/
├── README.md                 # 入口，先读这里
├── CLAUDE.md                 # vibecoding 工作纪律
├── docs/
│   ├── product-definition.md   # v0.3.1 封版产品定义（716 → 766 行）
│   ├── memory-architecture.md  # 技术架构（含 §7 个性化栈三件套）
│   ├── source-research.md      # 研究底稿（v0.1 起源）
│   ├── glossary.md             # 术语速查
│   ├── onboarding.md           # 你正在读
│   └── decisions/              # ADR (Architecture Decision Records)
├── roadmap/
│   └── milestones.md           # M0-M12 里程碑 + 三色判定 + 弹性规则
└── spikes/                     # 一次性破雷实验（M0 W1 ncps τ-bias）
    └── ncps_tau_bias/
```

---

## 关键术语（不要看不懂就问）

| 术语 | 一句话理解 | 详细 |
|------|-----------|------|
| LNN / LTC / CfC | 液态神经网络（连续时间 ODE，τ 可学）；本项目记忆系统的"心脏" | [glossary.md](glossary.md) §核心架构 |
| ncps | mlech26l/ncps，唯一成熟开源 LTC/CfC 实现（PyTorch backend） | [glossary.md](glossary.md) |
| 万能语法 20 槽 | 横切 L2-L5 的事件张量统一格式（P0/P1/P2/P3 四档） | [glossary.md](glossary.md) §万能语法 |
| EventTensor | 万能语法的张量数据结构 `[B, 20, 4]` | [memory-architecture.md](memory-architecture.md) §5.1 |
| 三件套 | v0.3.x 个性化栈 = τ 偏置 + FiLM + 槽位偏好向量（3KB/用户） | [memory-architecture.md](memory-architecture.md) §7 |
| CLS Consolidation | 海马-皮层双系统离线巩固（夜间 L2 → L3） | [memory-architecture.md](memory-architecture.md) §6 |
| vibecoding | 老板 + cc + codex 三方协作模式 | [CLAUDE.md](../CLAUDE.md) §2 |
| 三色判定 | 红/黄/绿早期预警，M1/M2/M3 末三次 | [milestones.md](../roadmap/milestones.md) §3 |
| ADR | Architecture Decision Records，每个重要决策一份 markdown | [decisions/](decisions/) |

---

## 你提的第一个 PR 怎么走

按 [`CLAUDE.md`](../CLAUDE.md) §3 + §4：

1. **接口先行**：写 Protocol（Python `typing.Protocol`）+ I/O schema
2. **测试先行**：单元测试定义验收标准（TDD：红 → 绿 → 重构）
3. **codex 对抗审查**：cc 写完立即调 codex 挑刺
4. **PR 描述模板**：[`CLAUDE.md`](../CLAUDE.md) §4 必填字段
5. **老板签名合并**：vibecoding 三道门最后一关

如果 codex 和 cc 给出**相反建议**：按 [`CLAUDE.md`](../CLAUDE.md) §3.1 流程走，并在 [`docs/decisions/`](decisions/) 留下 ADR。

---

## 红线（违反任意一条 → 立即停下问老板）

1. **不可妥协 6 项**（[`product-definition.md`](product-definition.md) §10.1）任一动摇
2. **三色判定连续两次红区**
3. **类型 3 数据使用边界灰色地带**
4. **节奏弹性触底**（M0 > 6w / M4 > 5.5m）
5. **未在当前对话内授权过的"对外操作"**（推送 GitHub / 发飞书 / 联系用户）

---

## 你有问题问谁

- **产品决策 / 路线问题** → 老板
- **代码实现 / 文档** → cc 主线
- **对抗审查 / 工程规范** → codex
- **调研 / review** → gemini（按需）
- **长时后台任务** → hermes（按需）

---

**版本：v0.1（M0 W1 骨架）· 2026-05-03**
**待 M0 末完整化：填充实际代码模块入口 + 第一个 PR 真实案例 + 调试技巧**
