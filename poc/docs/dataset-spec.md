---
name: PoC 数据集 Spec — 剧本 schema + 信号转换契约
audience: cc + codex（实现/审查）+ 老板（验收）
purpose: Day 3 数据生成框架的接口规范——ScriptEvent + Script JSON + SignalFrame JSONL + 报告
status: v0.1（2026-05-06 cc 起草）
parent: poc/docs/poc-spec.md
---

# PoC 数据集 Spec v0.1

## 0. 一句话目标

定义 LnMem PoC Day 3-4 数据生成的**输入输出契约**，确保以下三段 pipeline 互相能拼起来：

```
Persona (vboss_NNN.md)  →  Script (JSON)  →  SignalFrame stream (JSONL)  →  LNN training
   [picture taker]         [generate_scripts]   [script_to_signals]
```

下游 (Day 5+ LNN) 直接消费 JSONL 信号流——不准修改 Lane A 定义的契约。

---

## 1. ScriptEvent —— 剧本中的原子事件

定义在 [`poc/src/data/event_schema.py`](../src/data/event_schema.py)。

### 1.1 字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| event_id | str | ✅ | 全局唯一（建议 `<persona_id>_d<N>_e<NNN>`） |
| ts | str (ISO8601) | ✅ | 事件起始时间戳 |
| persona_id | str | ✅ | 与 Script.persona_id 一致 |
| day_index | int | ✅ | 与 Script.day_index 一致，0 起 |
| duration_s | float | ✅ | 范围 [5, 3600] |
| activity | Activity (enum) | ✅ | 8 类（详见 §1.2） |
| state | State (enum) | ✅ | 8 类（详见 §1.3，对齐 LNN StateRecognition 训练目标） |
| multimodal_hint | MultimodalHint | ✅ | 多模态画面提示（详见 §1.4） |
| description | str | ❌ | LLM 自由描述（≤ 30 字） |
| meta | dict | ❌ | 附加元数据 |

### 1.2 Activity 8 类

`work_focus / work_meeting / break_short / break_long / commute / leisure / sleep / transition`

> **Day 3 简化（codex F1-05 review 落档）**：`HeuristicLLMClient` 当前生成 8h 工作日剧本（09:00-17:00），不覆盖 `sleep` 类。Day 8+ 真实采集或 Day 4 真 LLM 接管时会补全，schema 保留 `sleep` 类是为了 Day 5+ LNN 训练接口一致。短期内 baseline 报告里 `sleep` 列为 0 属预期。

### 1.3 State 8 类

`low_energy / focused / distracted / fatigued / emotional_positive / emotional_negative / neutral / transitioning`

> 与 [`poc/docs/poc-spec.md`](poc-spec.md) §4 训练任务 1 + 主分支 `docs/capability-specs/cognitive.md` §3.3 对齐。

### 1.4 MultimodalHint —— 给 script_to_signals 的画面提示

| 字段 | 范围 | 含义 |
|------|------|------|
| eye_closure | [0,1] | 0=睁圆 / 1=闭眼；fatigued 态 ≥ 0.4 |
| head_pitch_norm | [0,1] | abs(head_pitch_deg) / 45（FusionConfig 上限） |
| head_yaw_norm | [0,1] | abs(head_yaw_deg) / 60 |
| gaze_stability | [0,1] | 0=游离 / 1=稳定注视 |
| voice_present | [0,1] | 0=无声 / 1=持续讲话 |
| utterance_density | [0,1] | 单段发言密度（÷ 4s 上限） |
| active_app | str | 前台 app 名；空串=未知 |
| app_switch_count_5min | int ≥ 0 | 5min 切换次数 |

下游 [`script_to_signals.py`](../src/script_to_signals.py) 把这 8 个字段还原成 [`fusion.py`](../src/fusion.py) 的 face/audio/context 三段输入字典，调 `MultimodalFusion.fuse()` 得到 12-dim FusionFrame。

---

## 2. Script JSON —— 单画像 × 单天剧本

文件路径：`poc/data/synthetic/script_<persona_id>_day<N>.json`

### 2.1 顶层结构

```json
{
  "persona_id": "vboss_001",
  "day_index": 0,
  "generated_at": "2026-05-06T08:30:00",
  "events": [ /* ScriptEvent.to_jsonl_dict() ... */ ],
  "meta": {
    "llm_class": "HeuristicLLMClient",
    "seed": 42,
    "target_hours": 8.0
  }
}
```

### 2.2 校验门槛

通过 [`script_validator.validate_script()`](../src/data/script_validator.py)（误差码见 §6）。

`ScriptValidationReport.passed == True` 才能落盘——这是 Day 3 的硬约束。

---

## 3. SignalFrame JSONL —— 信号流（LNN 输入）

文件路径：`poc/data/synthetic/signals_<persona_id>_day<N>.jsonl`

每行一个 JSON 对象：

```json
{
  "ts": "2026-05-06T09:00:03",
  "frame_idx": 12,
  "vector": [0.18, 0.0, 0.0, 0.85, 0.46, 0.59, 1.0, 0.0, 0.5, 0.5, 1.0, 0.7],
  "missing": [],
  "ground_truth_state": "focused",
  "activity": "work_focus",
  "persona_id": "vboss_001",
  "source_event_id": "vboss_001_d0_e002",
  "day_index": 0
}
```

### 3.1 vector 结构（与 [`fusion.py FEATURE_DIM=12`](../src/fusion.py) 对齐）

| 索引 | 含义 | 范围 |
|------|------|------|
| 0 | eye_closure | [0,1] |
| 1 | head_pitch_norm | [0,1] |
| 2 | head_yaw_norm | [0,1] |
| 3 | gaze_stability | [0,1] |
| 4 | fatigue_score | [0,1] |
| 5 | attention_score | [0,1] |
| 6 | pose_steadiness | [0,1] |
| 7 | blink_burst | {0, 1} |
| 8 | voice_present | {0, 1} |
| 9 | utterance_density | [0,1] |
| 10 | context_known | {0, 1} |
| 11 | focus_stability | [0,1] |

### 3.2 帧率与采样语义

PoC Day 3 默认 1Hz（与 fusion.py 取样契约一致）。Day 5+ LNN 训练可重采样 5/10Hz。

**采样区间约定（codex F3-01/F3-02 review 落档）**：左闭右开。`script_to_signals` 用
`math.ceil(duration_s * frame_rate_hz)` 帧数，第 i 帧时间 = 事件起 + `i * (1/hz)`。
即 5s × 1Hz 事件覆盖 t=[0, 4]s（5 帧），不含 t=5s。Day 5+ LNN 训练评估
要按此口径打标签。

### 3.3 噪声模型

高斯扰动 `N(0, sigma=0.05)` + clamp [0,1]，可重现（`random.Random(seed)`）。

---

## 4. 数据集报告 —— `dataset_report.py`

输入：N 个 Script JSON 路径 + 画像目录。

输出 `DatasetReport` + markdown：
1. 总览（N 画像 / N 剧本 / 总时长）
2. 状态分布每画像
3. 活动分布每画像
4. HEXACO 6 维两两欧式距离矩阵（标准 ≥ 0.4，与 [`data-generation-todo.md`](data-generation-todo.md) §4 #1 对齐）
5. 校验问题汇总

---

## 5. 端到端 pipeline 命令（Day 3 末验收）

```bash
cd /tmp/lnmem-fix/LnMem
source .venv-poc/bin/activate

# 1. 生成 3 画像 × 1 天剧本
python -c "
from pathlib import Path
from poc.src.generate_scripts import HeuristicLLMClient, run_batch
run_batch(
    persona_paths=[Path(f'poc/data/personas/vboss_00{i}.md') for i in (1,2,3)],
    day_index=0,
    llm=HeuristicLLMClient(),
    out_dir=Path('poc/data/synthetic'),
    seed=42,
)
"

# 2. 转 12-dim 信号
python -c "
from pathlib import Path
from poc.src.script_to_signals import script_file_to_signals_file
for i in (1,2,3):
    script_file_to_signals_file(
        Path(f'poc/data/synthetic/script_vboss_00{i}_day0.json'),
        Path(f'poc/data/synthetic/signals_vboss_00{i}_day0.jsonl'),
        seed=i,
    )
"

# 3. 出报告
python -c "
from pathlib import Path
from poc.src.dataset_report import build_report, write_markdown_report
report = build_report(
    sorted(Path('poc/data/synthetic').glob('script_*.json')),
    Path('poc/data/personas'),
)
write_markdown_report(report, Path('poc/data/synthetic/dataset_report.md'))
"
```

---

## 6. 错误码（script_validator）

| 严重 | 代码 | 含义 |
|------|------|------|
| error | EMPTY_EVENT_ID | event_id 空 |
| error | EMPTY_PERSONA_ID | persona_id 空 |
| error | BAD_TIMESTAMP | ts 不是 ISO8601 |
| error | DURATION_OUT_OF_RANGE | duration_s 不在 [5, 3600] |
| error | HINT_OUT_OF_RANGE | multimodal_hint 数值 ∉ [0,1] |
| error | NEGATIVE_APP_SWITCH | app_switch_count_5min < 0 |
| error | TIME_OVERLAP | 相邻事件时间重叠 |
| error | PERSONA_MISMATCH | 事件 persona_id 与 Script 不一致 |
| error | DAY_INDEX_MISMATCH | 事件 day_index 与 Script 不一致 |
| warning | LARGE_TIME_GAP | 相邻事件间隔 > 30s |
| warning | LOW_STATE_DIVERSITY | 出现 < 3 类状态 |

---

## 7. 与主分支 / 其他文档的归因

- [`poc/docs/poc-spec.md`](poc-spec.md) §2 Day 3 AC + §4 训练任务
- [`poc/docs/data-generation-todo.md`](data-generation-todo.md) §3 画像 schema + §4 评估标准
- [`docs/capability-specs/cognitive.md`](../../docs/capability-specs/cognitive.md) §5.2 多模态信号三表
- [`poc/src/fusion.py`](../src/fusion.py) FEATURE_DIM=12 协议（Day 2 落定，spec v0.1 文中"128 维"系起草时假想，已被实战 12 维取代）

署名：cc（Claude Opus 4.7）| 2026-05-06 v0.1
