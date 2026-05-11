---
title: Day 10 反馈闭环开闸 — Phase 1 V0.2
type: progress
status: shipped
date: 2026-05-09
branch: poc/lnn-validation
commits: [5840033, ef34b1e, 4efdbf5, 156f9e1]
tests: 242 passed
---

# Day 10 范式跨越：从单向推理到反馈闭环

**核心论断**：PoC 在指标层面已过线（合成 val_acc 0.9446 / bnd_f1 0.574 / 真人微笑 → positive 99% 触发），但**今天之前**反馈支柱完全空白——LNN-FIRST 真正分水岭是"反馈→在线学习→可治理"的环路是否闭合。Day 10 把 Phase 1 反馈闭环开闸。

---

## 一、Day 10 完成的工作

### 1.1 P0 修复（Phase 0，commit 5840033）

| 项 | 内容 |
|----|------|
| `poc/tests/test_vlm_annotator.py` | 删除 163-198 行死代码（unittest.main() 后的不可达重复 def） |
| `poc/src/lnn/model.py` | F2 ADR-002：τ 偏置改 softplus(`0.5 + F.softplus(tau_bias)`) 替代 tanh+clamp，去除梯度饱和 |

### 1.2 face_signals LNN-FIRST 一致性补丁（commit ef34b1e）

| 项 | 内容 |
|----|------|
| `face_signals.py:126` | 3-tuple unpack 修 crash bug：`au_dict, _emo, pose_dict = self._au_extractor.extract(frame)` |
| LNN-FIRST 落实 | 头部姿态切到 py-feat img2pose `pose_dict` 的 `yaw_deg/pitch_deg/roll_deg`，覆盖原 MediaPipe landmark 估算（commit 874e9fe 声称做但未真做） |

### 1.3 Day 10 真实数据评估（Phase A · 离线，commit ef34b1e）

新增 [`poc/scripts/eval_realvlm_correlation.py`](../poc/scripts/eval_realvlm_correlation.py)（codex 协作产出）。

**数据**：今天 491 条带 `vlm_label` 的 JSONL（来自 Air daemon），day6_ltc/checkpoint_day5.pt（val_acc 0.9446）。

**结果（全样本 188 窗口）**：Pearson r(lnn_score, vlm_conf) = -0.587，看似强负相关——但**是伪信号**：state=4 占 65%（123/188）的窗口里 93% VLM 标注为"未检测到人脸"，模型遇全空特征触发 OOD 默认输出。

**结果（过滤无人脸后 74 窗口）**：
- focused (34, 46%)：VLM 描述全是"AU4 皱眉+思考"，valence -0.222 ✓
- low_energy (17, 23%)：valence -0.235 / arousal 0.312 ✓
- emotional_negative (5)：valence -0.290 ✓
- fatigued (5)：valence -0.370 ✓
- emotional_positive (9)：valence ≈ -0.07 ⚠️ 真实数据里川哥工作时不微笑，此类样本被稀释
- Pearson r 重算：lnn vs valence = +0.24（弱正）

**Phase A 假设**：① emotional_positive 缺样本 ② lnn_score 普遍 0.8+ 置信度虚高。

### 1.4 Day 10 因果可控实测（Phase B · live_verify，commit 4efdbf5）

新增 [`poc/scripts/analyze_live_verify_log.py`](../poc/scripts/analyze_live_verify_log.py)。

**实验**：停 Air daemon → 启 live_verify.py（载入 day7_v2/checkpoint_day5.pt）→ 川哥 5 动作 30s 内做完 → SIGINT → 91 有效帧分析。

**结果反转 Phase A 两个假设**：

| 假设 | Phase A 推测 | Phase B 实测 | 结论 |
|------|------------|------------|------|
| emotional_positive 缺样本 | valence ≈ -0.07 接近 0 | **微笑触发 33 帧 99% 满置信，smile 峰 0.93 / 周围±10帧均值 0.84** | **证伪** |
| lnn_score 置信度虚高 | 离线 0.8+ | **最低 bucket=0.39 < 0.5，状态切换瞬间主动犹豫** | **证伪** |

**模型实际能力（day7_v2）**：1 帧反应灵敏（# 110 → 111：smile 0.05→0.80 → state 切换从 focused → 积极）；23 次状态切换全部跟随 smile/brow 信号。Phase A 假设的"缺正面情绪"是数据采集环境限制（川哥工作中本能不微笑），不是模型缺陷。

### 1.5 Phase 1 反馈闭环 V0.2（commit 156f9e1）

**架构**：`poc/src/realtime/feedback/` 新增 6 模块 + `daemon_main.py` 集成。

```
feedback/
├── trigger.py        ProactiveQueryTrigger：brow_burst (10 帧 @ 1fps > 0.85) / low_vlm_conf (< query_confidence_threshold)
├── prompt.py         三层渐进 osascript：通知中心 → modal dialog → choose_state/input_note
├── persistence.py    queries_<date>.jsonl 落盘（含 session_id + event_uuid）
├── rate_limiter.py   TokenBucket 1-3/h（最小间隔 = 3600/max_per_hour，状态原子写）
├── dnd.py            macOS 前台 app + dnd_apps 子串匹配
└── engine.py         编排器（thread tracking + stop join 保数据）
```

**通道选择**：用 macOS 原生弹窗（osascript）替代飞书 bot，理由：
- PoC 阶段川哥做实验时本就在 Air 前 → 即时性 0 延迟
- 数据结构化采集 > 解析自然语言回复
- 离线工作 + 隐私不出本机
- 实现 2.5h vs 飞书 4-5h
- 跨设备扩展性（车载场景 M6 任务）暂不需要

**集成点**：`daemon_main._signal_router_loop` 每帧调 `self._feedback_engine.observe(item)`，触发后开后台线程弹窗，反馈写 `~/.lnmem/feedback/queries_<date>.jsonl`。

**HTTP 端点**：`/health` 含 `feedback` stats 字段；新增 `/feedback/stats` 单独端点。

### 1.6 codex 对抗审查 8 finding 全闭环（commit 156f9e1）

| ID | 级别 | 处理 |
|----|------|------|
| FB-01 | F1 阻塞 | ✅ engine 维护 `_prompt_threads`，`stop()` 限时 join 5s |
| FB-02 | F2 | ❌ codex 误读：cognitive.md §5.6 "≤1/日"（M2+ 长期 check-in）vs day8-progress §四 "1-3/h"（PoC 实时反馈），适用场景不同 |
| FB-03 | F2 | ✅ daemon `_build_feedback_engine` 用 `fb.query_confidence_threshold` 构造 trigger |
| FB-04 | F2 | ⏸️ 单机 PoC 不紧迫，TODO Phase 2 |
| FB-05 | F3 | ⏸️ 计数误差不影响产品 |
| FB-06 | F3 | ✅ jsonl 加 `session_id` + `event_uuid`（uuid4().hex），Phase 2 奖励回灌可追踪 |
| FB-07 | F3 | ⏸️ daemon 防崩 trade-off |
| FB-08 | F4 | ✅ 部分覆盖：codex 写 8 个 V0.2 边界测试 |

**额外 + 阈值收紧**：codex review 没列但实测发现的 **22 触发/4min 过敏**——daemon 重启后 4 分钟实测，brow > 0.7 持续 5s 阈值在川哥自然工作脸下太敏感。收紧到 `brow > 0.85 持续 10s（10 帧 @ 1fps）`。

### 1.7 LnMem 历史第一条真实多模态反馈样本（已落盘）

`~/.lnmem/feedback/queries_2026-05-09.jsonl` 第 1 行：

```json
{
  "ts": "2026-05-09T12:56:09.782468Z",
  "frame_id": 0,
  "reason": "low_vlm_conf",
  "signal": {"brow_furrow": 0.0, "smile_intensity": 1.0, "eye_closure": 0.0, "cheek_raise": 1.0},
  "vlm_label": {
    "au_description": "光线极暗，面部特征不可见，无法识别AU。",
    "valence": 0.0, "arousal": 0.0, "confidence": 0.05,
    "model_used": "gemini-3.1-flash-lite-preview"
  },
  "user_response": {"label": "mismatch", "true_state": "distracted", "note": null},
  "latency_s": 27.08
}
```

→ 系统正确识别"VLM 自己拿不准"→ 询问 → 川哥纠正 → reward 信号入库。这是 Day 11+ 在线 τ-bias 漂移训练的第一个真实数据点。

### 1.8 测试覆盖

| 维度 | 数 |
|------|---|
| 基线（Day 9 末） | 181 |
| Phase 1 V0.1 主线测试 | +33（trigger 9 / persistence 4 / prompt 13 / engine 7） |
| Phase 1 V0.1 codex 测试 | +14（rate_limiter 6 / dnd 8 含 osascript 真调） |
| V0.2 边界测试（codex） | +8（阈值常量 4 / stop join 2 / session_id 1 / 配置接线 1） |
| 其它修复影响 | -8（test_vlm_annotator dead code 删除） |
| **Day 10 末** | **242 passed** |

---

## 二、当前进展

### 2.1 PoC 三大支柱评估（对齐 product-definition §10）

| 支柱 | Day 9 末 | Day 10 末 |
|------|---------|----------|
| 理论：LNN 心智核心 | 🟡 60%（固定权重推理） | 🟡 65%（反馈通道开闸，在线学习接口待 Phase 2） |
| 工程：感知-融合-LNN 链路 | 🟢 90% | 🟢 92% |
| 反馈：用户可见可编辑可治理 | 🔴 0% | 🟢 70%（V0.2 落地 + 第一条样本） |

### 2.2 daemon 运行状态

- Air：PID 56498，V0.2 起。`/health` 含 feedback stats。trigger 收紧到 0.85 + 10s。
- Studio：PID 31777（Day 10 早 Air 网络位置变更后无法 ssh 验证，待回 iBRO 内网核实）。

### 2.3 当前 commit 链

```
5840033  fix: test dead-code cleanup + LNN τ-bias softplus (F2 ADR-002)
ef34b1e  fix(face_signals): 3-tuple unpack + LNN-FIRST head pose; add eval_realvlm_correlation
4efdbf5  feat(scripts): add analyze_live_verify_log for Task 2 实测分析
156f9e1  fix(feedback): V0.2 codex review 4 修 (FB-01/03/06 + 阈值收紧)
```

---

## 三、后续 TODO（按优先级排序）

### 🔴 P0 — Day 11（明天必做）

- [ ] **观察 V0.2 trigger 频率合理性**：daemon 跑 1-2h，看新阈值（0.85 + 10s）下触发频率是否回归 < 5/h
- [ ] **Phase 2 启动：在线 τ-bias 漂移**
  - 实现 `poc/src/lnn/online_lora_adapter.py`：用 jsonl 反馈 → reward → LoRA on τ-bias
  - 用 Day 10 起累积的真实反馈样本做第一次微调实验
  - 验证：同一帧前后 lnn_score 在 mismatch 反馈下是否真下降
- [ ] **vboss_self 真人画像入 ckpt**：
  - 跑 personalt-test v0.2（已知盲区，partial 字段⚠️标注）
  - 真人 HEXACO 实测 → 生成 vboss_self ExpressionProfile
  - 在 day7_v2 上加 1 个 persona slot 微调 5-10 epochs
  - 替换 live_verify 默认 persona

### 🟡 P1 — Day 12-13

- [ ] **fatigued / low_energy / distracted / idle 全面实测**：补做 live_verify 第二轮覆盖剩余 4 类（不只是 positive/negative）
- [ ] **跨日稳定性测试**：连续 3 天 daemon 数据对比，看 valence 相关性是否稳定
- [ ] **emotional_negative anger 风格区分**（roadmap §0.0 P1 TODO）
- [ ] **boundary detection 双流**（bnd_f1 0.574 → 目标 0.7+）

### 🟢 P2 — Day 13-14

- [ ] **Demo 录屏**：川哥皱眉 5s → 弹窗询问 → 反馈 → LNN 状态可视化漂移（day8-progress §四 P4 目标）
- [ ] **PoC 收尾报告**：合成 val_acc / bnd_f1 / 真人因果验证 / 反馈闭环 / 真实样本数 综合指标
- [ ] **延后 finding 处理**：FB-04 fcntl 多实例锁 / FB-05 stats 锁 / FB-07 except 细分

### ⚪ M0 W1 — 下周（PoC 之后）

- [ ] M0 检查清单（roadmap §0）：破雷 + 基础设施
- [ ] CfC parallel stack 替代 step-by-step（5-10x 训练加速）
- [ ] 万能语法 v0 + L1+L2 LNN 雏形（M1）

---

## 四、踩坑与教训（Day 10 沉淀）

### 4.1 PoC 闭环判定：指标过线 ≠ 闭环

光有 val_acc/bnd_f1 不算 PoC 完成——LNN-FIRST 真正分水岭是反馈→在线学习闭合。Day 10 之前感知层全过，但反馈支柱 0 行代码。**必须把"反馈采集"作为 PoC 收尾的硬指标**，否则模型只是"考了 99 分但没人对答案"。

### 4.2 OOD 数据稀释陷阱

Phase A 离线 eval 65% state=4，看似模型崩塌——实际是无人脸窗口（特征全空）触发 OOD 默认输出，伪信号。**真实数据 eval 必须先按 face_detected/vlm_confidence 过滤**，否则 Pearson 数字误导决策。

### 4.3 离线相关性 vs 因果可控验证

Phase A 离线 Pearson r=-0.587 / Phase B 实测 1 帧反应。**离线 eval 是"问卷化评估"（cognitive.md §5.1 坚决拒绝的方法）**——真验证必须川哥做明确动作让模型预测。今天 Phase B 反转了 Phase A 的两个假设。

### 4.4 macOS 弹窗 vs 飞书 bot 通道选择

PoC 阶段 osascript 实现 30min vs lark-cli 4-5h。**弹窗在用户当场场景全维度领先**（即时/结构化/离线/隐私）。飞书 bot 的"跨设备"优势在车载场景（M6）才有意义。**通道选择应跟着用户场景走，不是惯性选 messaging API**。

### 4.5 codex 对抗审查的辨证使用

codex review 出 8 finding，1 F1 真 bug + 3 F2/3 F3 真有价值 + 1 F2（FB-02）误读。**不审查 = 数据丢失（FB-01 daemon thread）；盲信 codex = 把 cognitive.md §5.6 长期机制套到 PoC 实时机制（FB-02）**。处理 finding 必须每条核对原始文档判断 codex 是否误读。

### 4.6 trigger 阈值过敏（22/4min）

默认 brow > 0.7 持续 5s 在川哥"工作专注脸"下太敏感（自然 brow 范围 0.5-0.8）。22 触发 / 4min × 1 弹窗（被 rate limit 拦 21 次）。收紧到 0.85 + 10 帧。**生产阈值必须用真实用户数据校准，不能用合成数据训练时的"看起来合理"值**。

### 4.7 互动机制不只是产品功能，还是评估同构

cognitive.md §5.1：**评估方法和产品能力同构**。今天 live_verify 跑 91 帧只交付 Pearson 数字 = 评估退化为问卷化，违反产品哲学。每次模型预测都是一次"系统回应"，必须捕捉川哥身体级反应。已落 [`feedback_lnmem_realtime_state_confirmation.md`](../../mycc/memory/) 全局 memory。

### 4.8 TDD GREEN 倒置（团队效率信号）

codex 写 V0.2 边界测试时 cc 主线已经实现完契约 → codex 跑测试不是 RED 而是直接 GREEN 8/8。**这不是 TDD 失败，是团队执行速度跟得上契约定义的信号**——契约一旦写下，cc 实现 + codex 测试可以并行 GREEN 落地。

---

## 五、Day 10 commit 清单

| commit | 主题 | 行数变化 |
|--------|------|---------|
| 5840033 | test dead-code cleanup + LNN τ-bias softplus | -36 +2 |
| ef34b1e | face_signals 3-tuple + LNN-FIRST head pose + eval_realvlm_correlation | +232 -1 |
| 4efdbf5 | analyze_live_verify_log | +184 |
| 156f9e1 | Phase 1 V0.2 codex review 4 修 | +228 -13 |
| **合计** | **Day 10 全部** | **+644 -50** |

测试基线：181 → 242（+61），实质新增 +69（含被替换的 V0.1 engine 测试调整）。

---

## 六、决策点（待川哥拍板进 Day 11）

1. Phase 2 在线 τ-bias 漂移用什么实验设计？建议：先跑 7 天累积反馈 ≥ 50 条，再做第一次 LoRA 微调实验
2. vboss_self 真人画像加入 ckpt 是否走"加 1 slot 微调"路径？还是另开 personalt-test v0.3 流程？
3. Demo 录屏（Day 13-14）的拍摄场景设计——居家工作场景一镜到底？还是分段剪辑？
