---
name: LnMem PoC LNN 验证项目规范
audience: cc（核心执行）+ codex（对抗审查）+ 老板（每日 review）
purpose: 12-14 天 PoC 完整规范——目标/AC/DoD/时间盒/失败兜底/迁移协议
status: v0.1（2026-05-05 cc 起草，待 Day 0 老板 review 后定 v1.0）
parent: ../README.md
references:
  - ../../docs/principles/lnn-first.md（**P-1 最高准则 — 必读**）
  - ../../docs/product-definition.md（v0.3.3 主产品定义）
  - ../../docs/capability-specs/cognitive.md（v0.1.1 认知能力规范）
  - ../../docs/development-harness.md（v0.1 BRO-PDCA × 目标驱动验收）
  - ../../CLAUDE.md §2.1（PR 三道门 — 实际位置 §2.1，原 prompt 引用 §11.2 系归因错位）
---

# LnMem PoC LNN 验证项目规范 v0.1

> **⚠️ P-1 准则强制声明**：本 PoC 所有模块（capture/fusion/lnn/memory）必须遵守 [`../../docs/principles/lnn-first.md`](../../docs/principles/lnn-first.md)——记忆系统内核以 LNN（LTC/CfC 微分方程）为核心，凡"时间/演化/记忆/个性化"禁止用 LLM/规则替代 LTC。老板 2026-05-08 钦定。

## 0. 一句话目标

12-14 天工程时间 + 7 天老板自用，用一个完整的 PoC 证明：**LNN 真的能在多模态时序数据上学到主人特征**，并且这个学习过程**对老板和团队是可见、可解释、可治理的**。

---

## 1. 4 个不可妥协原则（违反就是 PoC 失败）

| # | 原则 | 反模式 |
|---|------|--------|
| 1 | **诚实优于惊艳** | 演示给团队的"AI 学习了你"，必须明说 LNN 30-40% / LLM 60-70% 的协作占比 |
| 2 | **架构形态优于性能数字** | 不追求 90%+ 准确率，追求"完整流程跑通 + 学习趋势可见 + 个性化参数分化" |
| 3 | **可解释优于黑箱** | LNN 训练 / 个性化参数 / 长期记忆 — 全部可视化 |
| 4 | **用户参与优于自动化** | 所有"AI 对主人的新理解"必须老板确认才进长期记忆（对应主分支 product-definition.md §10.1 第 6 项产品不可妥协） |

---

## 2. 时间盒 + 各阶段 AC（Acceptance Criteria）+ DoD（Definition of Done）

> **铁律**：Day 1 / Day 3 / Day 6 / Day 12 末四个硬门槛任一不达标 → 暂停推进 → 立即和老板讨论砍场景或延期。

### Day 0（老板 + cc 并行）

| 谁 | 任务 | DoD |
|---|------|-----|
| 老板 | 30-50 个虚拟主人画像 | `poc/data/personas/` 下至少 30 个 JSON 通过 cc 校验 |
| cc | 起草本规范 + data-generation-todo + limitations + migration-guide v0.1 骨架 + 校验脚本 | poc/docs/ 5 份文档 + poc/src/validate_personas.py 通过老板 review |

### Day 1-2：基础设施 + 多模态接入

**AC**：
- streamlit-webrtc 摄像头/麦克风可在 Mac M Air 稳定 30fps + 16kHz 采集
- MediaPipe Face Mesh 输出 468 个 landmarks，每秒 30 帧
- Whisper.cpp small（中文）本地推理可用，2 秒以内返回文本
- macOS 上下文捕获：当前 frontmost app + 窗口标题，每秒 1 次
- 信号融合输出：每秒 1 个 128 维特征向量（视觉 64 + 音频 32 + 上下文 32）

**Day 1 末硬门槛**：摄像头/麦克风/macOS 上下文信号能稳定流入 Python，每秒 1 个特征向量。**不达标 → 停下来排坑，不进 Day 2**。

**DoD**：
- `poc/src/multimodal/` 下三个模块（vision.py, audio.py, context.py）单独可跑
- `poc/src/fusion.py` 输出 128 维特征流
- 录制 10 秒样本，存盘 `poc/demos/day1_baseline.mp4`

### Day 3：数据生成框架

**AC**：
- 老板 Day 0 提供的 30 画像通过 schema 校验
- LLM 剧本生成 pipeline：每个画像 × 每天 8 小时 → JSON 时间序列
- 剧本到多模态信号转换：JSON 时间序列 → 128 维特征向量序列

**Day 3 末硬门槛**：必须有老板提供的虚拟主人画像 JSON。**老板未提供 → cc 用 5 个临时 mock 推进，但 Day 7 前必须替换为正式画像**。

**DoD**：3 个画像 × 1 天剧本完整跑通 + 数据集校验报告

### Day 4：批量生成虚拟数据

**AC**：30 画像 × 30 天 × 8h/天 = 7,200 小时合成多模态数据
**DoD**：数据质量验证报告（多样性 / 合理性 / 信号分布）

### Day 5-6：LNN 模型 + 虚拟数据预训练

**AC**：
- StateRecognitionLNN 在虚拟数据上准确率 60-75%
- BoundaryDetectionLNN F1 > 0.5
- 不同虚拟主人个性化参数明显分化（t-SNE 可视化能看到聚类）

**Day 6 末硬门槛**：状态识别 < 30% / 模型发散 → **停下来检讨架构，不进 Day 7**。失败兜底：简化为仅"状态识别"单任务（去掉 boundary + 个性化），保留至少一条可演示的 LNN 学习证据。

**DoD**：训练 checkpoint + t-SNE 可视化 + 训练 loss 曲线

### Day 7：5 阶段记忆形成流程

**AC**：从实时多模态信号到结构化长期记忆，5 个阶段全部跑通：
1. **重要性标记**（毫秒级）：启发式规则输出 [0,1] importance score
2. **工作记忆整合**（秒-分钟）：CfC LNN 持续运行 + deque(maxlen=20) 高重要性事件缓冲
3. **事件分割**（分钟-小时）：4 个触发条件 — 长停顿 / app 切换 / 主动关键词 / 30 分钟时间窗
4. **精炼与索引**（小时-天）：LLM 槽位补全 + 情感标注 + 候选区 48h TTL
5. **Consolidation**（夜间或主动）：4-Part 报告 + 用户确认机制

**DoD**：完整流程在虚拟数据上跑通，UI 5 个面板全部可见

### Day 8-12：老板自用 + 累积真实数据

**AC**：
- PoC 部署稳定，老板每天用 4-6 小时
- 累积 25-30 小时真实多模态数据
- 每晚自动 consolidation 跑通

**Day 12 末硬门槛**：真实数据 < 15 小时 → 延长老板自用 1-3 天。失败兜底（O-Q3）：**老板对 PoC 体验严重不满 → 立即停 + 复盘根因 + 决定是否重启**。

**DoD**：`poc/data/real/` 累积日志 + 每日 retro 在 `poc/CHANGELOG.md` 落档

### Day 13：真实数据微调

**AC**：
- 加载 Day 6 预训练 checkpoint
- 用老板真实数据微调个性化参数（τ 偏置 + FiLM）
- 评估对比：预训练 vs 微调后准确率 / 老板参数与虚拟主人参数空间距离

**DoD**：对比报告 + 微调 checkpoint

### Day 14：演示设计 + 录屏

**演示对象**：内部团队（含老板）— O-Q2 答案

**AC**（5 个核心可视化全部出片）：
1. 状态识别准确率曲线（启动 vs 7 天后）
2. 个性化参数热力图（τ 偏置 + FiLM gamma/beta + 槽位偏好）
3. t-SNE 老板 vs 30 虚拟主人参数空间分布
4. 启用 / 不启用 LNN 学习的消融对比
5. AI 学习日志面板实时录屏

**DoD**：
- `poc/demos/` 下 3 个工作场景录屏 + 5 个可视化录屏
- 至少 1 个"AI 识别错被纠正"或"AI 选择不介入"的真实瞬间
- `poc/docs/limitations.md` 完整诚实声明
- `poc/docs/migration-guide.md` "PoC → 主分支工程化" 完整建议

---

## 3. 3 个工作场景（演示用）

| 场景 | 触发 | 期望 AI 行为 | 演示价值 |
|------|------|------------|---------|
| 1 工作疲惫的非语言识别 | 多模态信号：瘫坐 + 揉眼 + 叹气，无语音 | 右下角极小提示"累了？走走 5 分钟" | 克制比主动重要 |
| 2 工作中断恢复 | 老板 40 分钟前说要重构，刚回来问"我刚才在想啥" | 引用具体上下文（编辑文件 + 行号 + 当时表达） | ChatGPT 完全做不到的痛点 |
| 3 低置信度澄清 | 老板说"我妈让我周末回家"，平淡表情 + 正常语速 | "记下了。需要我做什么吗？"（不擅自共情） | 场景感知 + 留白设计 |

---

## 4. 3 个 LNN 训练任务

| 任务 | 输入 | 输出 | 训练目标 |
|------|------|------|---------|
| 1 状态识别 | 128 维多模态特征序列 | 8 类离散状态（low_energy/focused/distracted/fatigued/emotional_positive/emotional_negative/neutral/transitioning） | 交叉熵 + 时间一致性 |
| 2 事件边界检测 | 连续时间序列 | 二分类（边界 / 非边界） | BCE + 真值边界 |
| 3 个性化参数学习 | per-user τ 偏置（64 维）+ FiLM gamma/beta（128×2）+ 槽位偏好（20 维） | 共享 LNN 上的个性化调制 | 不同用户参数空间分化（contrastive） |

详细模型架构与代码骨架在 [`poc/src/lnn/`](../src/lnn/)（Day 5 创建）。

---

## 5. 5 阶段记忆形成机制

完整规范见 [docs/capability-specs/cognitive.md](../../docs/capability-specs/cognitive.md) §3.3 + §5（认知能力规范 v0.1.1，主分支封版）。

PoC 实现位置：[`poc/src/memory/`](../src/memory/)（Day 7 创建），不简化、不偷工。

**关键不可省略点**（HARNESS-CHECK-NO-SKIP 铁律）：
- 阶段 5 Consolidation 的 4-Part 用户确认机制必须完整实现（不允许"简化为自动写入"）
- 用户确认 3 个按钮（确认 / 先观察 / 删除）必须全部可点击 + 行为正确
- 删除按钮必须实际触发"加入黑名单"逻辑，不允许只是 UI 假象

---

## 6. 失败兜底标准（O-Q3 答案明示化）

PoC 哪些情况下应**主动告知"PoC 阶段性失败"**：

| 信号 | 判定 |
|------|------|
| Day 1 末多模态信号流入未达标 + 1 天排坑无果 | 停 + 复盘技术栈选型 |
| Day 6 末 LNN 训练发散 / 状态识别 < 30% | 停 + 评估架构简化路径 |
| Day 12 末真实数据 < 10 小时（不是 15h 而是 10h 这个更严的门槛） | 停 + 重新评估老板自用强度 |
| **老板对 PoC 体验严重不满**（明确卡 4-6h/天用机会让 cc 知道） | **立即停 + 根因复盘 + 决定是否重启** |
| 主分支工程化的可迁移性 < 50% | 停 + 重新评估 PoC 设计 |

---

## 7. 资源边界（O-Q4 + O-Q5 答案落档）

| 资源 | 边界 | 应急方案 |
|------|------|---------|
| 算力（LNN 8-16h 训练） | MacBook Air M 主路径 | M2 Max Studio 备路径 / 云端 GPU 租赁 |
| LLM API（Day 4 数据生成 1000+ 调用 + Day 7-12 每晚 consolidation） | 主路径走 Claude Opus 4.7 + DeepSeek-V4 备 | 老板直接补购预算 |
| codex 配额（cc 主导 + codex 对抗审查每模块 1 次） | 4-6 个模块 × 1 次审查 = 6 次 | 配额不足时老板补购 |

---

## 8. vibecoding 协作模式

按 [CLAUDE.md §2.1](../../CLAUDE.md) PR 三道门：

1. **cc 实现** — 写代码、写测试、写文档
2. **codex 对抗审查** — 找最容易翻车的 5 个点
3. **老板 review + 决定是否进下一天**

每日 retro 写入 [`poc/CHANGELOG.md`](../CHANGELOG.md)。

---

## 9. 与主分支的归因（grep 验证）

| 主分支文档 / 章节 | 引用内容 | grep 验证 |
|------------------|---------|----------|
| [docs/product-definition.md](../../docs/product-definition.md) §10.1 第 6 项 | "用户可见、可编辑、可治理的记忆界面"产品不可妥协 | ✅ line 584 命中 |
| [docs/product-definition.md](../../docs/product-definition.md) §1.3 第 5 条 | "AI 学习你从黑箱变成对账单"产品承诺 | ✅ line 41 命中 |
| [docs/product-definition.md](../../docs/product-definition.md) §13 双门 | MVP α 验证标准（PoC 复用此双门设计） | ✅ line 677 命中 |
| [CLAUDE.md](../../CLAUDE.md) §2.1 | PR 三道门 + 日 retro | ⚠️ 原 prompt 引用 §11.2 不准，实际 §2.1（line 35） |
| [docs/capability-specs/cognitive.md](../../docs/capability-specs/cognitive.md) §3.3 + §5 | 5 阶段记忆形成 + 多模态反馈信号 RL 评估 | ✅（v0.1.1 主分支封版） |
| [docs/development-harness.md](../../docs/development-harness.md) | BRO-PDCA × 目标驱动验收（PoC 严格遵守） | ✅（v0.1 主分支封版） |

---

## 10. PoC → 主分支工程化迁移协议

**两阶段流程**（HARNESS 纪律）：
- 阶段 A（本 PoC）：在 `poc/lnn-validation` 长期分支验证
- 阶段 B（PoC 验证后）：**不直接合并**，而是把 PoC 验证产出（数据 schema / 模型架构 / 5 阶段记忆设计）作为输入，**在主分支按工程标准重新实现**

详细迁移建议见 [`migration-guide.md`](migration-guide.md)（Day 14 末完整版出）。

**PoC 不修改主分支文档**（除非发现致命错误，需先讨论）。

---

## 11. 文档维护协议

- 本规范是 PoC 的"宪法"——所有 PoC 决策都参考本文档
- 每日 retro 时审视本规范是否需要修订
- 修订记入 [`poc/CHANGELOG.md`](../CHANGELOG.md)
- 同步更新 v0.1 → v0.X，不允许悄悄改

署名：cc（Claude Opus 4.7）| 2026-05-05 起草 v0.1
