---
name: PoC 虚拟主人画像设计 Brief（老板 Day 0 立即用）
audience: 老板（亲自设计画像）+ cc（验证、批量生成数据）
purpose: 给老板提供 Day 0 一份可立即开干的画像设计指南——schema + 评估标准 + 范例 + 提交协议
status: v0.1（2026-05-05 cc 起草）
parent: poc/docs/poc-spec.md
---

# PoC 虚拟主人画像设计 Brief

> **谁读这份文档**：老板（必读）、cc（用来批量生成数据 + 验证画像质量）
> **读完后老板能立即做的事**：打开 `poc/data/personas/` 目录，按 schema 写第 1 个 JSON
> **预算**：1-2 天，30-50 个画像
> **阻塞关系**：老板 Day 0 不完成 → cc Day 4 数据生成无法启动 → 整个 PoC 推迟

---

## 1. 一句话目标

为 PoC LNN 训练造**两两差异显著、覆盖典型职业类型、含极端案例**的 30-50 个虚拟主人画像，作为合成数据剧本生成的输入。

---

## 2. 为什么这一步老板必须亲做（不是 cc 代劳）

| 维度 | 理由 |
|------|------|
| **决定 PoC 成败** | 画像质量决定 LNN 训练质量——画像太相似 → LNN 学不到差异 → 个性化机制无法验证 |
| **产品哲学的具体化** | 画像本身在回答"什么样的人值得被 AI 理解"——这是 cc 不能替老板决定的事 |
| **t-SNE 验证的锚点** | 后续 t-SNE 可视化需要"和老板自己画像高度相似的 2-3 个虚拟主人"作为聚类锚——老板自己画像未在 PoC 内显式写出，靠"相似画像"间接验证 |
| **诚实声明的源头** | PoC 完成后 limitations.md 要写"30 虚拟主人不能覆盖全人口"——老板自己造的画像才能诚实知道它的盲区在哪 |

---

## 3. 画像 Schema（完整字段）

每个画像一个 JSON 文件，命名 `vboss_NNN.json`（NNN = 001-050），存放路径 `poc/data/personas/`。

```json
{
  "id": "vboss_001",
  "type_label": "深度工作型程序员 / 外向产品经理 / 创业者 / ...",
  "demographics": {
    "age_range": "30-40",
    "gender": "male / female / non-binary / prefer-not-to-say",
    "profession": "...",
    "city_tier": "tier1 / tier2 / tier3",
    "language_primary": "zh / en / mixed"
  },

  "traits": {
    "focus_duration_minutes": [90, 120],
    "fatigue_pattern": "下午 3-4 点必现 / 全天均匀 / 晚上反而清醒 / ...",
    "interruption_tolerance": "high / medium / low",
    "emotion_baseline": "活跃 / 平静 / 低落 / 起伏大",
    "stress_triggers": ["DDL 临近", "意外打扰", "多任务切换", "..."],
    "communication_style": "直接 / 委婉 / 学术化 / 口语化",
    "work_style": "深度型 / 切换型 / 协作型 / 独立型",
    "rest_pattern": "番茄钟 / 工作累了就停 / 不主动休息 / ...",
    "tech_savviness": "expert / intermediate / novice"
  },

  "ai_interaction_preferences": {
    "during_work": "极简提示 / 精准回答 / 被动响应 / 主动建议",
    "after_work": "陪伴对话 / 任务清单 / 不打扰",
    "deep_night": "完全不打扰 / 轻量提醒 / 安静监听",
    "preferred_response_length": "1-3 句 / 1-2 段 / 详细解释",
    "tolerates_proactive_suggestion": true
  },

  "key_distinguishing_features": [
    "和其他虚拟主人最核心的差异点 1（必须具体到行为，不是形容词）",
    "差异点 2",
    "差异点 3",
    "差异点 4（可选）",
    "差异点 5（可选）"
  ],

  "expected_multimodal_signals": {
    "facial_expression_baseline": "微表情少 / 表情丰富 / 抿嘴 / 皱眉为主 / ...",
    "voice_pace_baseline": "慢语速 0.8x / 标准 1.0x / 快语速 1.2x",
    "voice_tone_baseline": "低沉 / 平稳 / 高亢 / 起伏大",
    "typical_posture": "前倾专注 / 后仰放松 / 频繁起身 / 长时间不动"
  },

  "notes_for_data_generation": "（可选）特殊指示给 LLM 剧本生成器，比如'此画像偶尔会自言自语，工作中插入与项目无关的轻声评论'"
}
```

### 字段说明

- `id`：必须 `vboss_NNN`，N=3 位数字，方便排序和引用
- `type_label`：一句话标签，3-10 字
- `traits`：核心区分维度，**至少 3 个 traits 与其他画像显著不同**
- `key_distinguishing_features`：3-5 条具体行为描述（不是形容词），cc 用来检查画像差异化
- `expected_multimodal_signals`：用来约束 LLM 剧本生成的多模态信号——**这一节是 PoC 训练成败的关键**

---

## 4. 评估标准（cc 收画像后会跑一遍）

| # | 标准 | 不达标的后果 | cc 检查方式 |
|---|------|-------------|------------|
| 1 | 30-50 个画像两两至少 3 个 traits 显著不同 | LNN 个性化参数学不到差异 | 自动两两比对 + 输出差异矩阵 |
| 2 | 至少 5 种典型职业类型 | 数据多样性不足 | 按 `type_label` 分组统计 |
| 3 | 含极端案例：极度内向 / 极度外向 / 极度焦虑 / 极度平和（至少 1 个/类） | 边界场景验证缺失 | grep `key_distinguishing_features` |
| 4 | 至少 2-3 个画像与老板自己画像高度相似（用于 t-SNE 验证） | t-SNE 锚点缺失 | 老板自己标记 `similar_to_self: true` |
| 5 | `expected_multimodal_signals` 全部填写（不允许 null） | LLM 剧本生成无约束，多模态信号失真 | JSON schema 强制校验 |

---

## 5. 操作流程

### Day 0 老板侧

1. 打开 `poc/data/personas/` 目录
2. 复制本文档 §6 提供的 3 个范例（`vboss_001.json` / `vboss_002.json` / `vboss_003.json`）作为起点
3. 按 schema 填写到第 30 个画像（先冲量再优化）
4. **每 10 个画像后通知 cc 做一次轻量验证**（避免到第 30 个才发现差异化不足重做）
5. 完成 30 个后，检查极端案例覆盖（4 类至少各 1 个）
6. 根据 PoC Day 4 数据多样性反馈，决定是否补到 50 个

### cc 侧（Day 0 同步）

1. 起草 schema 校验脚本 `poc/src/validate_personas.py`（轻量 JSON schema + 差异化矩阵打分）
2. 老板每提交 10 个画像，cc 跑一次校验，输出报告：
   - 每个画像的差异化得分（与其他画像的最近邻距离）
   - 覆盖度报告（职业 / 极端案例 / 年龄 / 性别）
   - 不达标画像列表 + 修改建议

---

## 6. 范例（3 个完整画像，老板可直接 copy 当起点）

> 这 3 个范例是 cc 起草的"起点模板"，老板**可以全部改写**——不要被范例局限。范例只是为了证明 schema 可填、字段含义清楚。

### 范例 1：vboss_001.json — 深度工作型程序员

```json
{
  "id": "vboss_001",
  "type_label": "深度工作型程序员",
  "demographics": {
    "age_range": "32-38",
    "gender": "male",
    "profession": "后端架构师",
    "city_tier": "tier1",
    "language_primary": "zh"
  },
  "traits": {
    "focus_duration_minutes": [90, 150],
    "fatigue_pattern": "下午 3-4 点眼皮沉，但不主动休息，靠咖啡续命",
    "interruption_tolerance": "low",
    "emotion_baseline": "平静",
    "stress_triggers": ["生产 bug 半夜被叫起", "需求改来改去", "代码 review 被反复要求重写"],
    "communication_style": "直接",
    "work_style": "深度型",
    "rest_pattern": "工作累了就停，但停不超过 3 分钟",
    "tech_savviness": "expert"
  },
  "ai_interaction_preferences": {
    "during_work": "极简提示",
    "after_work": "任务清单",
    "deep_night": "完全不打扰",
    "preferred_response_length": "1-3 句",
    "tolerates_proactive_suggestion": false
  },
  "key_distinguishing_features": [
    "工作时长时间不动，姿势前倾，眼睛盯屏幕不眨",
    "讲话语速快、句子短、信息密度高",
    "不喜欢被打断——任何主动提示都会先标记为骚扰",
    "疲惫时会无意识揉眼睛、摸鼻梁、长叹气（但不停下工作）"
  ],
  "expected_multimodal_signals": {
    "facial_expression_baseline": "微表情少，专注时眉头微锁",
    "voice_pace_baseline": "快语速 1.2x",
    "voice_tone_baseline": "平稳偏低",
    "typical_posture": "前倾专注，长时间不动"
  },
  "notes_for_data_generation": "此画像在工作中几乎不说话，多模态信号以视觉 + 上下文为主；偶尔会对屏幕骂一句脏话（轻声）"
}
```

### 范例 2：vboss_002.json — 外向产品经理

```json
{
  "id": "vboss_002",
  "type_label": "外向产品经理",
  "demographics": {
    "age_range": "28-35",
    "gender": "female",
    "profession": "B 端产品经理",
    "city_tier": "tier1",
    "language_primary": "zh"
  },
  "traits": {
    "focus_duration_minutes": [25, 45],
    "fatigue_pattern": "全天均匀，但情绪起伏大",
    "interruption_tolerance": "high",
    "emotion_baseline": "活跃",
    "stress_triggers": ["跨部门对齐", "研发说做不了", "用户反馈差"],
    "communication_style": "委婉但快速决策",
    "work_style": "切换型",
    "rest_pattern": "番茄钟严格 25/5",
    "tech_savviness": "intermediate"
  },
  "ai_interaction_preferences": {
    "during_work": "主动建议",
    "after_work": "陪伴对话",
    "deep_night": "轻量提醒",
    "preferred_response_length": "1-2 段",
    "tolerates_proactive_suggestion": true
  },
  "key_distinguishing_features": [
    "工作时频繁切换应用：飞书/Figma/邮件/IDE 平均 5 分钟切一次",
    "经常自言自语推演用户场景",
    "情绪表达直接：开心时笑出声，烦躁时叹气加挠头",
    "每 25 分钟番茄钟必起身倒水或拉伸"
  ],
  "expected_multimodal_signals": {
    "facial_expression_baseline": "表情丰富，眉毛动作多",
    "voice_pace_baseline": "标准 1.0x",
    "voice_tone_baseline": "起伏大",
    "typical_posture": "频繁起身"
  },
  "notes_for_data_generation": "此画像音频信号丰富，含大量自言自语、电话会议、跨部门对齐；视觉信号也丰富（表情起伏大）"
}
```

### 范例 3：vboss_003.json — 焦虑创业者（极端案例）

```json
{
  "id": "vboss_003",
  "type_label": "焦虑型早期创业者",
  "demographics": {
    "age_range": "30-40",
    "gender": "male",
    "profession": "AI 创业者（CEO + CTO 双角色）",
    "city_tier": "tier1",
    "language_primary": "mixed"
  },
  "traits": {
    "focus_duration_minutes": [40, 80],
    "fatigue_pattern": "晚上 9-11 点反而清醒（焦虑驱动），凌晨崩溃",
    "interruption_tolerance": "low",
    "emotion_baseline": "起伏大（焦虑 + 短暂兴奋）",
    "stress_triggers": ["融资进度", "团队离职", "竞品发版", "现金流"],
    "communication_style": "直接但情绪化",
    "work_style": "深度型 + 切换型混合",
    "rest_pattern": "不主动休息，疲惫到崩溃才停",
    "tech_savviness": "expert"
  },
  "ai_interaction_preferences": {
    "during_work": "精准回答 + 偶尔主动建议",
    "after_work": "陪伴对话（深夜需要倾诉）",
    "deep_night": "安静监听 + 紧急时轻量提醒",
    "preferred_response_length": "1-2 段",
    "tolerates_proactive_suggestion": true
  },
  "key_distinguishing_features": [
    "工作时频繁查看手机（融资进展 / 团队消息）",
    "深夜会突然进入低落状态，自言自语 30 秒以上",
    "讲话夹杂中英文，技术名词 + 商业术语混用",
    "焦虑时会无意识抖腿、摸下巴、来回拨头发",
    "周末和工作日完全没区别（工作时长一样）"
  ],
  "expected_multimodal_signals": {
    "facial_expression_baseline": "起伏大，皱眉 + 短暂大笑交替",
    "voice_pace_baseline": "起伏大（专注时快、低落时慢）",
    "voice_tone_baseline": "起伏大",
    "typical_posture": "频繁小动作（抖腿/摸下巴）"
  },
  "notes_for_data_generation": "此画像是极端案例（焦虑型）。深夜剧本必须包含 1-2 段独白式自言自语，含负面情绪信号（叹气/低声抱怨）；同时有少量正面爆发（融资进展好的瞬间）"
}
```

---

## 7. 命名 / 路径 / 提交约定

| 项 | 规则 |
|----|------|
| 文件命名 | `vboss_NNN.json`（NNN=001-050，3 位数字） |
| 路径 | `poc/data/personas/vboss_NNN.json` |
| 编码 | UTF-8（中文不转义） |
| 提交节奏 | 每完成 10 个 → cc 验证 → 通过后下一批 |
| 通知 cc 方式 | 飞书消息或直接说"画像 vboss_001~010 写完了，cc 验证一下" |
| 修改协议 | 老板提供方向，cc 不擅自改老板写的画像（除明显 schema 不合规） |

---

## 8. cc Day 0 同步任务（老板做画像时 cc 在做什么）

| 任务 | 输出 | 进度回报节奏 |
|------|------|------------|
| 起草 `validate_personas.py` 校验脚本 | `poc/src/validate_personas.py` | Day 0 末 |
| 起草数据生成 pipeline 骨架（不开始批量调用 LLM） | `poc/src/generate_scripts.py` v0.1 | Day 1 |
| 起草 `poc-spec.md` AC + DoD | `poc/docs/poc-spec.md` | Day 0 末 |
| 起草 `limitations.md` 骨架 | `poc/docs/limitations.md` | Day 0 |
| 整理 Whisper.cpp / MediaPipe / streamlit-webrtc 依赖清单 | `poc/requirements-poc.txt` | Day 1 |
| Day 1 末多模态信号流入硬门槛准备 | 摄像头 / 麦克风 / macOS 上下文捕获脚本骨架 | Day 1 |

---

## 9. 与主分支的归因

本文档参考主分支以下章节（`/tmp/lnmem-fix/LnMem` 现状）：

- [docs/product-definition.md](../../docs/product-definition.md) §6 冷启动数据策略 — 合成数据多模态信号要求
- [docs/product-definition.md](../../docs/product-definition.md) §10.1 不可妥协 6 项第 4 项 — per-user 个性化栈（PoC 验证 τ 偏置 + FiLM）
- [docs/capability-specs/cognitive.md](../../docs/capability-specs/cognitive.md) §5.2 强正/强负/弱信号三表 — `expected_multimodal_signals` 字段设计依据

---

**老板 Day 0 立刻能做的 4 步**：

1. `cd /tmp/lnmem-fix/LnMem/poc/data/personas/`
2. 把范例 §6 的 3 个 JSON 复制下来（保留 / 修改 / 替换均可）
3. 按 schema 写到第 10 个，通知 cc 做轻量验证
4. 通过后冲到第 30 个

**有阻塞或不清楚的地方随时找 cc。**

署名：cc（Claude Opus 4.7）| 2026-05-05 起草 v0.1
