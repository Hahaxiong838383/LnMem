# LnMem · 以 LNN 为心脏的记忆系统

> **如果你是 Claude Code / Codex 第一次到这里**：跳到 [给 AI 同事的 30 秒入口](#给-ai-同事的-30-秒入口)。
> **如果你是工程师**：读 [快速开始](#快速开始)。
> **如果你是投资人/老板**：读 [`docs/product-definition.md`](docs/product-definition.md) §0 §1。

---

## 它是什么

**一个能"陪你数年"的 AI 伙伴的记忆系统**——居家+车载持续运行，用万能语法把你说的、看到的、做的事统一编码，用液态神经网络（LNN）作为永远在线的心智核心。

**与传统"LLM + RAG"方案的本质区别**：传统方案在对话时检索过去；LnMem 在永远在线的连续时间状态里持有当下，通过事件分割主动写入，通过夜间巩固主动学习。

**项目阶段**：v0.3.2 产品定义微调 · M0 W2 启动准备中（2026-05-04）

> **v0.3.2 微调（2026-05-04）**：基于外部独立技术分析的工程信号吸收——CfC 升级为 LNN 主路径（推理速度 ↑ 1-5 数量级、LTC 仅作研究对比）、Liquid-S4 升级为 τ≈小时反思层候选主路径、LNN 多模态训练风险下调 🟡→🟢、新增 v0.4 探索清单。**不是新方向，是外部验证已有方向后的精确化**。详见 [`docs/product-definition.md`](docs/product-definition.md) §14.0。

---

## 给 AI 同事的 30 秒入口

> **本仓库对所有 AI coding 工具开放协作（Claude Code / Codex / Gemini CLI / Cursor / Copilot / 等）**——但任何 AI 工具读取本项目时**必须先遵循 [`docs/STYLE.md`](docs/STYLE.md) 的写作规范**，否则产出会被拒收。详见各 AI 工具的入口文件：[`CLAUDE.md`](CLAUDE.md) / [`AGENTS.md`](AGENTS.md) / [`GEMINI.md`](GEMINI.md) / [`.cursorrules`](.cursorrules) / [`.github/copilot-instructions.md`](.github/copilot-instructions.md)。

文档读取顺序（**按这个顺序读，不要跳序**）：

1. [`docs/STYLE.md`](docs/STYLE.md) — **所有 .md 写作规范，强制**（4 级体裁分层 + L1 frontmatter + L2 万能公式 v1）
2. [`docs/development-harness.md`](docs/development-harness.md) — **所有开发任务工程纪律，强制**（BRO-PDCA × 目标驱动验收：O 阶段书面化 AC + DoD / C 阶段按 AC + DoD 逐项验收 + 附证据）
3. [`docs/capability-specs/cognitive.md`](docs/capability-specs/cognitive.md) — **认知能力规范**，涉及意图预测 / 多模态信号 / 评估机制必读（评估方法和产品能力同构，拒绝问卷/弹窗/评分）
4. [`docs/product-definition.md`](docs/product-definition.md) §0–§1（30 秒愿景）
5. [`docs/memory-architecture.md`](docs/memory-architecture.md) §0–§2（30 秒技术骨架）
6. [`CLAUDE.md`](CLAUDE.md)（vibecoding 工作纪律，**必读**）
7. [`roadmap/milestones.md`](roadmap/milestones.md) §当前月份段（你现在该做什么）

**红线规则**：
- 任何对 [§10 不可妥协 6 项](docs/product-definition.md) 的妥协建议——**先停下问老板**
- 三色判定（[`roadmap/milestones.md`](roadmap/milestones.md)）连续两次进入红区——**先停下问老板**
- 类型 3 数据（团队真实交互）使用边界灰色地带——**先停下问老板**
- **任何 .md 文档不带 L1 frontmatter 直接 commit**——CI 应当拒绝（M0 W3 加 pre-commit hook）
- **任何开发任务跳过 O 阶段 AC + DoD 直接动手 / C 阶段美化"未达成"为"基本完成"**——红线违规（详见 [`docs/development-harness.md`](docs/development-harness.md) §1 铁律）

**提交规则**：
- 所有 PR 必须经过 cc 实现 + codex 对抗审查 + 人类工程师签名
- AI 不是责任主体（[`CLAUDE.md`](CLAUDE.md) §AI 同事责任边界）
- 文档改动必须遵循 [`docs/STYLE.md`](docs/STYLE.md) 分级规范

---

## 文档地图

```
LnMem/
├── README.md                    # 你正在读
├── CLAUDE.md                    # vibecoding SOP + AI 同事工作纪律
├── docs/
│   ├── product-definition.md    # 产品定义 v0.3.2（封版，819 行）
│   ├── memory-architecture.md   # LNN-centric 记忆系统技术架构 v0.1
│   ├── source-research.md       # 12/18 月可行性研究底稿（含 v0.3.2 §3.4 Liquid-S4 升级）
│   ├── lnn-math.md              # LNN 数学方程实战手册（LTC + CfC + 个性化三件套通俗注解）
│   ├── glossary.md              # 关键术语字典
│   └── decisions/               # ADR 记录（001 路线 B / 002 M0 骨架对抗审查）
└── roadmap/
    └── milestones.md            # M0-M12 里程碑 + 红黄绿三色 + M0 启动清单
```

---

## 核心架构（一图）

```
[多模态感知] → [万能语法 Parser] → [EventTensor 20 槽]
                                          ↓
                              [L1 LNN τ≈秒, 64 neurons]
                                          ↓
                              [L2 LNN τ≈分钟, 32 neurons]
                                          ↓
                              [事件分割门控] —boundary?→
                                          ↓
                                  [L2 情景记忆 · Graphiti KG]
                                          ↓ (夜间)
                                  [CLS Consolidation]
                                          ↓
                                  [L3 语义/人格记忆 · LoRA + 偏好向量]
                                          ↓
                                  [今天 / 我对你的新理解 / 记忆账本]
                                          ↓
                                  [用户可读/编/删]
```

### 分层详解（作用 / 技术原理 / 场景举例）

> 三栏并排：第一栏让人秒懂"这层是干什么的"；第二栏给技术决策点便于挑漏洞；第三栏给"如果没有这层，用户会发生什么"，反向证明这层非做不可。

**0 · 多模态感知层**
- **作用**：把摄像头画面、麦克风音频、车机/家居 IoT 信号、用户输入文字这些异构原料统一成"模态嵌入 + 时间戳"流，喂给上层。
- **技术原理**：图像走 Qwen3-VL 7B 取 CLS token、音频走 Fun-ASR 出文本 + Qwen-Audio 出语音特征、文本走 Qwen3 Embedding。所有模态都被对齐到 1024 维 + ms 级时间戳，对齐方式是按时间窗口 concat。瓶颈点：多模态时间对齐误差 > 200ms 会让上层的事件分割误判。
- **场景**：你坐进车里说"放点轻松的"——视觉拿到"驾驶座面孔+疲惫表情"、音频拿到"人声+语义=放松音乐"、IoT 拿到"刚启动+车速 0"，三路合成一个 t=14:32:01 的多模态包，上送 Parser。
- **如果没有这层**：上层只能看到"放点轻松的"这句话，丢失"疲惫"这个关键修饰，建议会变成爆裂音乐。

**1 · 万能语法 Parser → EventTensor 20 槽**
- **作用**：把多模态包压缩成 20 槽固定 schema 的事件张量——这是横切 L2-L5 的统一格式，等于给整个系统装了"标准件"。
- **技术原理**：20 槽分 4 优先级（P0 必填 / P1 检索主键 / P2 时态推理 / P3 仅向量化），用一个 BERT-style encoder + 20 个 slot-specific head 抽取，confidence 用 head 输出的 softmax 最大值。槽位填不下的长尾叙事走 Overflow Capsule 自由文本通道。瓶颈点：槽位不全时（例 P0 < 60%）下游 LNN 训练会发散，所以 M1 末有红区门槛。
- **场景**：刚才那个多模态包被 Parser 拆成：`who=主人`、`when=2026-05-03 14:32`、`where=车里`、`mood=疲惫`、`intent=放松`、`object=音乐`、`modality=车机+人脸`...20 个槽位填进去 18 个，剩下 2 个空槽位 + 1 个 Overflow（"今天有点不想说话"）。
- **如果没有这层**：每层用各自的格式，L2-L5 之间数据互相不认识，整个系统散架，无法跨层做"上周三晚上和谁吃饭"这种跨时间检索。

**2 · L1 LNN（τ≈秒，64 neurons）— 工作记忆**
- **作用**：承载"现在正在发生什么"——秒级敏感度，捕获瞬时情绪、意图、注意力。**它就是 AI 的"当下意识"**。
- **技术原理**：用 ncps 库的 LTC (Liquid Time-Constant)，每个神经元有可学的时间常数 τ，通过常微分方程 `dx/dt = -x/τ + f(input, state)` 演化，PyTorch backend。64 neurons 是经验值（参考 ncps 论文 + L1 任务复杂度估算）。瓶颈点：LTC 训练对学习率敏感，可能发散——所以 M1 末有"训练完全发散 → 红区"的明确门槛。
- **场景**：你说话的语速从平稳变急促，L1 的隐状态 0.3 秒内就变了，下游能感知到"主人激动了"——而传统 RNN/Transformer 是离散步进的，必须等下一个 token 才更新，慢一拍。
- **如果没有这层**：AI 的"当下感"消失，会像传统聊天机器人一样"等你说完我才反应"，无法在你刚皱眉时就调整语气。

**3 · L2 LNN（τ≈分钟，32 neurons）— 短期叙事**
- **作用**：承载"这一小段时间在发生什么"——比 L1 更慢的演化，对应"一通对话""一段路程""一顿饭"这种自然事件单元。
- **技术原理**：CfC (Closed-form Continuous-time)，比 LTC 推理快 2-3×（适合长时间在线运行）。32 neurons，τ ≈ 60s。L1 → L2 是 LNN 的纵向堆叠（state stacking），不是简单的池化。瓶颈点：双层 LNN 联合训练比单层难，需要梯度裁剪 + warmup。
- **场景**：你跟 AI 聊"周末想去哪玩"，L1 在每句话间感知情绪；L2 在 5 分钟对话结束时，状态向量收敛到"主人想要安静的、近郊的、不需要做攻略的地方"——这就是 L2 的产物，是后面事件分割和写入 KG 的基础。
- **如果没有这层**：L1 直接接事件分割，会因为太敏感而把"一通对话"切成 50 个事件，KG 被噪声淹没。

**4 · 事件分割门控（boundary gate）**
- **作用**：判断"刚才这段是不是一个完整事件，要不要写入长期记忆"——决定 KG 写入频率和颗粒度。
- **技术原理**：v0 是规则版（输入沉默 > 30s / 物理位置切换 / 主语切换 / 槽位变更率突变 任一触发），v1 升级为学习版（用 L2 状态变化率 + 当前 EventTensor 与历史平均的距离训练一个二分类器）。瓶颈点：F1 < 0.3 → 红区，因为这个直接决定下游记忆质量。M2 末第一次评估这个指标。
- **场景**：你跟 AI 聊完音乐推荐，沉默了 1 分钟，然后说"对了，明天提醒我交房租"——门控在沉默处触发 boundary，前一段"音乐推荐"作为完整 episode 写入 L2 KG，后一段"房租提醒"是新 episode。
- **如果没有这层**：要么写得太少（KG 一天就 3 条记忆），要么写得太多（KG 一天 500 条全是片段噪声），都会让"上周三晚上"这种检索失效。

**5 · L2 情景记忆（Graphiti KG，bi-temporal）— 长期可检索记忆**
- **作用**：以"何时-何地-与谁-做了什么"为单元的事件库——支持时间锚点检索。**这是"它会记得"承诺的物理载体**。
- **技术原理**：getzep/graphiti，bi-temporal 双时间索引（t_valid = 事件何时发生 / t_invalid = 何时被新事实推翻）。中文 entity 抽取用 GLiNER-multilingual 替代默认英文模型。每个节点 = 一个 EventTensor 实例化的 episode，边 = 实体共现/因果。瓶颈点：中文 entity F1 实测 < 0.5 → 黄区，需要老板亲自标 100 条对照集。
- **场景**：你三个月后问 "上周三晚上和谁吃饭"——KG 用 t_valid 锚定到 "2026-08-13 19:30"，用 entity edge 拉出 `who=同事老张`，返回完整 episode：`{when=8/13 19:30, where=金鼎轩, with=老张, mood=放松, topic=他孩子升学}`。
- **如果没有这层**：AI 只能用最近几轮对话回答你，"上周三"这种问题直接答不出，违反核心承诺 1。

**6 · CLS Consolidation（夜间巩固，海马-皮层双系统）**
- **作用**：每天凌晨 3 点把 L2 的事件库提炼成 L3 的语义/偏好 —— **这是"它会变化"承诺的引擎**。
- **技术原理**：受脑科学 Complementary Learning Systems 启发——海马快学（L2 当天事件全留）+ 皮层慢学（L3 用 LoRA 增量微调 + 偏好向量 EMA 更新）。算法：从 L2 取昨天所有 episode，按"重要度 × 频率 × 情绪强度"排序取 Top-K（K = 50），用 LLM 蒸馏成 "新偏好/新规则/新关系" 三类，写入 L3。瓶颈点：合成数据污染——如果 LoRA 学到的是 LLM 编的偏好而不是真实主人偏好，L3 会越学越歪。所以 M1 红区有"合成数据 pipeline 跑不通"门槛。
- **场景**：连续一周你在晚 11 点都说"困了"或"明早要提醒我"，CLS 在第 7 天夜里固化成 L3 偏好向量 `bedtime_anchor=23:00 ± 30min, sleep_intent_strength=0.85`——次日 22:50 你再开口，AI 会主动问"要准备睡了吗"。
- **如果没有这层**：AI 永远是"今天的它"，第 100 天仍不比第 1 天更懂你，违反核心承诺 2。

**7 · L3 语义/人格记忆（LNN 原生三件套）— 长期人格画像**
- **作用**：承载"主人是怎样的人"——周-年时间尺度的偏好、习惯、价值观、关系网络。
- **技术原理**：v0.3.x **LNN 原生轻量个性化三件套**（替代原 LoRA 方案，3KB/用户，可 CPU 推理）：
  1. **Per-user τ 偏置（< 1KB/用户）**：在 LTC 时间常数上加 per-user 偏置向量 Δτ_user，决定"什么信息留多久"——LTC 原论文（Hasani et al. 2021）中 τ 本就可学，加 per-user 偏置是其自然延伸，不破坏 ODE 形式 `dx/dt = -x/(τ+Δτ_user) + f(input, state)`
  2. **输入侧 FiLM 调制（~2KB/用户）**：`f(γ_user · x + β_user, state)`，等价于"用户特化的输入透镜"，不动内部状态演化方程；冷启动 γ=1, β=0（即"无个性化"），渐进展开
  3. **万能语法槽位偏好向量（80 字节/用户）**：直接注入 LNN 输入维度，和 v0.3 §1.4 核心 IP 完美契合（之前 LoRA 路径其实没用上这个 IP）

  **LLM 语言风格个性化**：路径 1（MVP α）= 偏好向量序列化作为 system prompt 注入 / 路径 2（M7+ 评估后）= 极轻量 LoRA rank=2

  **资源对比**：原 LoRA 方案 50MB/用户 + 单卡 24GB 微调 + 10 用户排队 → 新方案 3KB/用户 + CPU 可跑 + 10K 用户同服务器无压力

  **变更轨迹**：v0.1 否决 State-offset Tuning（直接加在状态 x 上破坏 ODE）→ v0.3 改 LoRA + 偏好向量 + BitFit（规模化路径有空白）→ **v0.3.x 改 τ 偏置 + FiLM + 偏好向量**（数学结构兼容 + 规模化路径打开）。τ 偏置作用在衰减常数、FiLM 作用在输入侧，都不破坏 ODE，区别于 v0.1 的 State-offset。

  **瓶颈点 / 待校验项**（M2-M3 必须验证，进三色判定）：① τ 偏置 < 1KB/用户的容量上限——做容量曲线（1KB / 4KB / 16KB 行为区分度边际收益），不够则回退到 4KB；② FiLM × LTC 联合训练稳定性——warmup + 梯度裁剪 + γ/β 安全初始化；③ system prompt 注入的 token 成本盈亏点（M7+ 时 vs LoRA rank=2 比较）
- **场景**：你是程序员、不爱社交、对效率敏感、晚睡——τ 偏置让"工作截止日期"这类信息在你的 LTC 状态里衰减更慢（你确实更在意），FiLM 让"早起 / 集体活动"这类输入被压低权重，偏好向量让 P0 槽位倾向于"效率"和"独处"。AI 推荐"周末活动"时，自动避开"集体活动 / 早起 / 户外暴晒"。
- **如果没有这层**：每个用户的 AI 都长一个样，违反"用一个月后的它比第一天更懂你"承诺。
- **潜在学术贡献**：LNN 学术界目前几乎没人做"在物理参数层（τ）做个性化"。M2-M3 容量曲线如果对数收敛（小预算大收益），M6 启动论文准备《Per-user τ-Bias: Lightweight Personalization for Liquid Time-Constant Networks》。

**8 · 用户可见层（今天 / 我对你的新理解 / 记忆账本）**
- **作用**：把黑箱的 L2/L3 翻译成用户可读、可编辑、可治理的 UI——**这是"它知道边界 / 它会让你看见"两条承诺的载体**。
- **技术原理**：「今天」= 今早抽 L2 昨日 Top-3 episode；「我对你的新理解」= CLS 昨夜对 L3 的 diff（哪些偏好变强/减弱）；「记忆账本」= 全量记忆变更日志（增/合/删）+ 用户可点删除按钮（24h 软删除可恢复）。瓶颈点：UI 不能让用户感觉"在审计 AI"，必须像看朋友圈，仪式感设计是 v0.4 《惊喜时刻设计手册》的核心。
- **场景**：早上你打开 App 看到"今天：1) 昨晚跟老张吃饭聊他孩子升学 2) 你提到周末想去近郊 3) 提醒：明天要交房租"——你点掉第 2 条说"我后来反悔了不去近郊"，KG 立刻把这条 episode 软删除，CLS 下次巩固不会再用它。
- **如果没有这层**：AI 是黑箱，用户无法纠错，错误会复利累积，违反核心承诺 4 + 5。**这一项是不可妥协 6 项里的产品 §6**。

详见 [`docs/memory-architecture.md`](docs/memory-architecture.md) §2 - §7。

---

## 核心承诺（v0.3 §1.3）

1. **它会记得**：能回答"上周三晚上和谁吃饭"等含时间锚点的查询
2. **它会变化**：用一个月后的它，比第一天更懂你
3. **它在场景间无缝**：早晨家里、下午车里、晚上回家——同一个它
4. **它知道边界**：你能看到、能改、能删、能带走
5. **它会让你看见**：黑箱学习 → 对账单（行业首创）

---

## 不可妥协的 6 项（v0.3 §10.1）

**技术**：
1. 万能语法 20 槽作为全系统共享 schema
2. 多时间尺度 LNN 心智核心（哪怕只有两层）
3. CLS 双系统离线 consolidation 流程

**产品**：
4. per-user 个性化栈（**v0.3.x 更新**：LNN 原生三件套 = τ 偏置 + FiLM + 槽位偏好向量，3KB/用户。LLM 语言风格走 system prompt 注入 / M7+ 评估后启动 LoRA rank=2）
5. 居家+车载双场景统一架构
6. 用户可见、可编辑、可治理的记忆界面

---

## 时间线（vibecoding 提速版 · v1.1 含弹性规则）

| 月 | 关键里程碑 | 默认 | 允许弹性 |
|----|-----------|------|---------|
| M0 | 基础设施 + 万能语法 v0 + ncps spike + working slice | 4 周 | +2 周（至 6 周封顶） |
| M1–M3 | LNN 雏形 → 事件分割 → 个性化 v0 | 各 1 月 | 累计 +3 周 |
| **M4** | **🎯 MVP α**：种子用户开始日均使用 | 5 月末 | +0.5 月（**5.5 月封顶**） |
| M5–M6 | CLS 巩固 + 车载场景试点 | — | M4 验证后再排 |
| **M7-M8** | **🎯 MVP β**：扩到 25–50 用户 | — | 同上 |
| M9–M11 | 主动推理 + 多设备同步 + 长期 KG 治理 | — | 同上 |
| M12 | 18 月架构成型（提前 6 月达成） | — | 同上 |

> **节奏哲学**（v1.1）：激进时间线≠每个里程碑按时，是每个交付物被认真对待。M4 超过 5.5 月必须停下来全盘重评，不许默默拖延。详见 [`roadmap/milestones.md`](roadmap/milestones.md) §3 弹性规则。

红黄绿三色判定见 [`roadmap/milestones.md`](roadmap/milestones.md)。

---

## 快速开始（M0 启动后填充）

```bash
# M0 第一周完成后这一节会更新
git clone https://github.com/Hahaxiong838383/LnMem.git
cd LnMem
# pip install -r requirements.txt   # 待 M0 第一周建立
# python -m synthetic_data_generator # 待 M0 第一周建立
```

详见 [`roadmap/milestones.md`](roadmap/milestones.md) M0 启动检查清单。

---

## 风险已知（v0.3 §9）

🔴 **高**：国产 LLM API 限流 / Graphiti 中文 entity 质量 / 用户付费意愿未知 / 合成数据污染 LNN 训练
🟡 **中**：事件分割长尾 / 车载集成成本
🟢 **中**（v0.3.2 下调）：LNN 多模态训练稳定性（CfC 主路径解决 ODE solver 不稳，但 ncps PEFT 空白与训练数据稀缺仍是真问题）

每个风险都有降级路径，详见 [`docs/product-definition.md`](docs/product-definition.md) §9。

---

## 协作链路（v1.1 · 单人 + AI 协作模式）

> **路线 B 决定**（详见 [`docs/decisions/001-route-b-single-founder-ai-mode.md`](docs/decisions/001-route-b-single-founder-ai-mode.md)）：M0-M3 单人启动 + AI 三方协作；M3 末看效果再决定是否扩团队。这是基于老板 mycc/EvoDesk/brain 已有的工作模式延伸。

| 角色 | 职责 |
|------|------|
| **老板** | 产品决策 / 路线判断 / 用户访谈 / vibecoding 三道门最终签名 / "第 0 号用户"（M1 末启动） |
| **Claude Code（cc）** | 架构设计 / 代码主线 / 文档 / 测试 / 周报整理 |
| **Codex（gpt-5.4）** | 对抗审查 / 工程规范 / 边界条件 |
| **Gemini**（按需） | 调研 / review（不写代码） |
| **Hermes**（按需） | 长时后台任务 |

**硬件**：Air（开发）/ Mini（编排+网关）/ Studio M2 Max 64GB（本地训练 LTC/CfC + 推理）/ NAS（文件归档），均已就位。
**LLM API**：DeepSeek + Qwen + GLM + Claude + GPT-5.4 五家 quota 已就位。
**M3 扩团队决策点**：v0.3.x 三件套（3KB/用户、CPU 可推理）大幅降低单人扛得住的概率。M3 末按真实工作量数据决策。

详见 [`CLAUDE.md`](CLAUDE.md) + [`docs/decisions/`](docs/decisions/)。

---

**v0.3.2 微调（M0 启动用版本，外部技术分析工程信号吸收） · 2026-05-04**
