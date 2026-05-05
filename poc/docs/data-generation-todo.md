---
name: PoC 虚拟主人画像设计 Brief（老板 Day 0 立即用）
audience: 老板（亲自设计画像）+ cc（验证、批量生成数据）
purpose: 给老板提供 Day 0 一份可立即开干的画像设计指南——七层 schema + 评估标准 + 范例 + 提交协议
status: v0.2（2026-05-05 老板提供范例后升级 schema 至七层 HEXACO 结构）
parent: poc/docs/poc-spec.md
---

# PoC 虚拟主人画像设计 Brief

> **谁读这份文档**：老板（必读）、cc（用来批量生成数据 + 验证画像质量）
> **读完后老板能立即做的事**：参考 [`vboss_001`](../data/personas/vboss_001.md) / [`vboss_002`](../data/personas/vboss_002.md) / [`vboss_003`](../data/personas/vboss_003.md)，写第 4 个画像
> **预算**：1-2 天，30-50 个画像
> **阻塞关系**：老板 Day 0 不完成 → cc Day 4 数据生成无法启动 → 整个 PoC 推迟

---

## 1. 一句话目标

为 PoC LNN 训练造**两两差异显著、覆盖典型职业类型、含极端案例**的 30-50 个虚拟主人画像，作为合成数据剧本生成的输入。

**真实分布锚点**（老板用大型样本研究确定）：
- **依恋分布**：安全型 ~63.5% / 回避型 ~22.2% / 焦虑型 ~5.5% / 恐惧-回避 15-20%
- **HEXACO 高智商人群信号**：高 H + 高 C + 低 E（情绪性）

画像设计**必须贴近真实人群分布**，不是极端虚构。

---

## 2. 为什么这一步老板必须亲做（不是 cc 代劳）

| 维度 | 理由 |
|------|------|
| **决定 PoC 成败** | 画像质量决定 LNN 训练质量——画像太相似 → LNN 学不到差异 → 个性化机制无法验证 |
| **产品哲学的具体化** | 画像本身在回答"什么样的人值得被 AI 理解"——这是 cc 不能替老板决定的事 |
| **t-SNE 验证的锚点** | 后续 t-SNE 可视化需要"和老板自己画像高度相似的 2-3 个虚拟主人"作为聚类锚——老板自己画像未在 PoC 内显式写出，靠"相似画像"间接验证 |
| **诚实声明的源头** | PoC 完成后 limitations.md 要写"30 虚拟主人不能覆盖全人口"——老板自己造的画像才能诚实知道它的盲区在哪 |

---

## 3. 画像 Schema（七层结构 + 顶部 frontmatter）

每个画像一个 markdown 文件，命名 `vboss_NNN.md`（NNN = 001-050），存放路径 `poc/data/personas/`。

**结构**：YAML frontmatter（cc 可机读）+ Layer 1-7 markdown 描述（老板自然书写）

### 3.1 Frontmatter（必填，cc 校验脚本读取）

```yaml
---
id: vboss_NNN
name: 中文姓名
type_label: 一句话标签（3-12 字）
mbti_tag: INTJ / ENFJ / INFP / ...  # 仅作快速沟通用，不进训练

hexaco:
  honesty_humility: 0.0-1.0
  emotionality: 0.0-1.0
  extraversion: 0.0-1.0
  agreeableness: 0.0-1.0
  conscientiousness: 0.0-1.0
  openness: 0.0-1.0

attachment:
  anxiety: 0.0-1.0
  avoidance: 0.0-1.0
  type: secure / anxious / avoidant / fearful_avoidant / secure_with_anxious_tendency
  population_match: "X%"  # 该类型在大型样本中的人群占比

motivation:
  self_efficacy: 0.0-1.0
  autonomy_need: 0.0-1.0
  competence_need: 0.0-1.0
  relatedness_need: 0.0-1.0

values_top3: [Schwartz 10 价值观中选 3 个]
values_bottom: [Schwartz 10 价值观中选 1-3 个]

demographics:
  age_range: [N, M]
  gender: male / female / non-binary
  profession: 简短描述
  city: 城市
  family: 简短描述
  commute_minutes: 数字 / 0

similar_to_self: false  # 老板可标记 true 用于 t-SNE 锚点
ai_trust_build_speed: slow / fast / nonlinear
multimodal_signal_density: low / medium / high / extreme_bipolar
---
```

**Schwartz 10 价值观清单**（供 values_top3/bottom 选择）：
`autonomy / achievement / hedonism / stimulation / self-direction / universalism / benevolence / tradition / conformity / security / power`

### 3.2 Body 七层（markdown 自由书写）

完整模板见 [`vboss_001.md`](../data/personas/vboss_001.md)。七层包括：

1. **Layer 1：基础特质**（HEXACO 6 维 + 关键 facets + 表现描述）
2. **Layer 2：依恋模式**（焦虑 + 回避维度 + 类型 + 关系表现 + 对 AI 互动含义）
3. **Layer 3：动机层**（自我效能 / 自主 / 能力 / 关系需求 + Schwartz 价值观 top3/bottom）
4. **Layer 4：if-then 情境签名**（≥ 5 个情境，每个含行为 / 情绪 / 多模态三段）
5. **Layer 5：身份与情境**（demographics + 当前生活章节 + 核心张力）
6. **Layer 6：多模态特征签名**（视觉基线 + 疲劳/压力/专注信号 + 语音基线 + 行为基线）
7. **Layer 7：AI 交互产品设计**（信任建立路径 + 沟通风格 + 避免话题 / 特别注意）

---

## 4. 评估标准（cc 收画像后自动跑）

| # | 标准 | 不达标的后果 | cc 检查方式 |
|---|------|-------------|------------|
| 1 | **HEXACO 6 维余弦距离 ≥ 0.4**（两两） | LNN 个性化参数学不到差异 | frontmatter `hexaco` 6 维向量两两余弦计算，输出差异化矩阵 |
| 2 | **依恋分布合理**：安全型不超过 65%、含 ≥ 2 个回避 + ≥ 1 个焦虑 + ≥ 1 个恐惧-回避 | 训练时极端依恋类型欠拟合 | frontmatter `attachment.type` 计数 |
| 3 | **至少 5 种典型职业类型** | 数据多样性不足 | frontmatter `demographics.profession` 分组统计 |
| 4 | **极端案例覆盖**（极内向 / 极外向 / 高 N / 高 O 各 ≥ 1） | 边界场景验证缺失 | hexaco 单维 ≤ 0.30 或 ≥ 0.85 计数 |
| 5 | **至少 2-3 个画像 `similar_to_self: true`** | t-SNE 锚点缺失 | frontmatter 字段统计 |
| 6 | **Layer 4 if-then 情境 ≥ 5 个**（每画像） | LLM 剧本生成约束不足 | parse 章节标题"情境 N" |
| 7 | **Layer 6 多模态信号 4 类信号全描述**（视觉/疲劳/压力/专注 + 语音 + 行为） | 多模态 ground truth 缺失 | parse 章节关键字 |

**单画像质量门**：以上 7 条任一不达标 → cc 标注"建议修订" → 老板回看修订。

**集合质量门**：30 画像收齐时跑总报告，含：
- HEXACO 6 维分布散点图
- 依恋类型饼图
- 职业 / 城市 / 年龄 / 性别覆盖度
- 两两余弦距离矩阵热力图

---

## 5. 操作流程

### Day 0 老板侧

1. 打开 `poc/data/personas/`，参考已有 [`vboss_001.md`](../data/personas/vboss_001.md) / [`vboss_002.md`](../data/personas/vboss_002.md) / [`vboss_003.md`](../data/personas/vboss_003.md)
2. 复制其中一个最接近的当起点，按七层 schema 写下一个画像
3. **每写完 5 个通知 cc 做一次轻量验证**（避免到第 30 个才发现差异化不足重做）
4. 完成 30 个后，检查极端案例覆盖（4 类至少各 1 个）
5. 根据 PoC Day 4 数据多样性反馈，决定是否补到 50 个

### cc Day 0 同步任务

1. 起草 schema 校验脚本 `poc/src/validate_personas.py`：
   - parse YAML frontmatter
   - 校验 7 条评估标准
   - 输出差异化矩阵 + 集合覆盖报告
2. 老板每提交 5 个画像 → cc 跑一次校验 + 反馈
3. Day 0 末输出 baseline 报告（仅用 cc 起草的 vboss_001~003 作为基线）

---

## 6. 范例（直接打开 3 份完整画像）

老板已提供 3 个完整范例，覆盖不同分布象限：

| ID | 标签 | HEXACO 高分 | HEXACO 低分 | 依恋 | 主导动机 | 链接 |
|----|------|-----------|------------|------|---------|------|
| vboss_001 | 内向高自律的技术管理者 | C (0.85) / O (0.70) / H (0.72) | E (0.30) | 安全偏回避 | 自主 + 能力 | [vboss_001.md](../data/personas/vboss_001.md) |
| vboss_002 | 高能量高同理心的产品负责人 | E (0.78) / A (0.72) | (无显著低) | 偏安全略焦虑 | 关系 + 成就 | [vboss_002.md](../data/personas/vboss_002.md) |
| vboss_003 | 高敏感的独立创作者 | O (0.92) / H (0.80) / Em (0.78) | E (0.25) | 恐惧-回避 | 自主 + 审美 | [vboss_003.md](../data/personas/vboss_003.md) |

**三两余弦距离**：HEXACO 6 维上两两 > 0.4 ✅（已达标）

**依恋分布覆盖**：3 个画像分别覆盖 22% / 60% / 17.5% 三大人群。

### 6.1 老板已点出的待补缺口（继续写画像时优先补）

| 缺口 | 大致比例 | 建议形态 |
|------|---------|---------|
| 极度焦虑型 | 5-19% | 高 N + 高焦虑维度 + 中 E（外向焦虑表现强） |
| 戏剧性高 E + 高 N | ~5% | 表演型，能量起伏大，剧烈正负情绪交替 |
| 老年画像（55+） | ~10% | 老板说"PoC 主要 25-45 岁可不覆盖"——可暂缓 |

---

## 7. 命名 / 路径 / 提交约定

| 项 | 规则 |
|----|------|
| 文件命名 | `vboss_NNN.md`（NNN=001-050，3 位数字） |
| 路径 | `poc/data/personas/vboss_NNN.md` |
| 编码 | UTF-8（中文不转义） |
| 提交节奏 | 每完成 5 个 → cc 验证 → 通过后下一批 |
| 通知 cc 方式 | 直接说"画像 vboss_001~005 写完了，cc 验证一下" |
| 修改协议 | 老板提供方向，cc 不擅自改老板写的画像（除明显 frontmatter schema 不合规） |

---

## 8. cc Day 0 同步任务（老板做画像时 cc 在做什么）

| 任务 | 输出 | 进度回报节奏 |
|------|------|------------|
| 起草 `validate_personas.py` 校验脚本 | `poc/src/validate_personas.py` | Day 0 末 |
| 起草 `requirements-poc.txt` 依赖清单 | `poc/requirements-poc.txt` | Day 0 末 |
| 起草数据生成 pipeline 骨架（不开始批量调用 LLM） | `poc/src/generate_scripts.py` v0.1 | Day 1 |
| 整理 Whisper.cpp / MediaPipe / streamlit-webrtc 依赖清单 | 同上 | Day 1 |
| Day 1 末多模态信号流入硬门槛准备 | 摄像头 / 麦克风 / macOS 上下文捕获脚本骨架 | Day 1 |

---

## 9. 与主分支的归因

本文档参考主分支以下章节（`/tmp/lnmem-fix/LnMem` 现状）：

- [docs/product-definition.md](../../docs/product-definition.md) §6 冷启动数据策略 — 合成数据多模态信号要求
- [docs/product-definition.md](../../docs/product-definition.md) §10.1 不可妥协 6 项第 4 项 — per-user 个性化栈（PoC 验证 τ 偏置 + FiLM）
- [docs/capability-specs/cognitive.md](../../docs/capability-specs/cognitive.md) §5.2 强正/强负/弱信号三表 — Layer 6 多模态特征签名设计依据

---

**老板 Day 0 立刻能做的 4 步**：

1. `cd /tmp/lnmem-fix/LnMem/poc/data/personas/`
2. 打开 [`vboss_001.md`](../data/personas/vboss_001.md) / [`vboss_002.md`](../data/personas/vboss_002.md) / [`vboss_003.md`](../data/personas/vboss_003.md) 任意一份当模板
3. 复制并改写，按七层 schema 写到第 5 个，通知 cc 验证
4. 通过后冲到第 30 个

**有阻塞或不清楚的地方随时找 cc。**

署名：cc（Claude Opus 4.7）| 2026-05-05 v0.2（老板提供范例后升级）
