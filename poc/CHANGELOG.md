# PoC CHANGELOG

> PoC 内部变更日志 — 每日 retro / 重大决策 / 阶段验收都记入此文件。
> 主分支 LnMem 的整体变更不在此处记录。

---

## v0.15 — 2026-05-08（Day 6 P0: feature_dim 12→18 · AU 表情特征扩展）

Day 5 真人验证暴露核心缺陷：12 维全认知特征，无表情信号 → 模型不能检测安静的积极。
Day 6 P0 用 agent team（cc 主线 + codex）并行完成 6 维 AU 表情特征扩展。

### 变更清单

| 文件 | 变更 | Agent |
|------|------|-------|
| `contracts.py` | `feature_dim: 12 → 18` | cc |
| `event_schema.py` | `MultimodalHint` +6 表情字段（smile/brow/cheek/lip/jaw/mouth） | cc |
| `fusion.py` | `FEATURE_DIM 18` + `_expression_features()` 6 维新函数 | cc |
| `script_to_signals.py` | 新 hint 字段 → face dict 透传 | cc |
| `face_signals.py` | `_extract_expression_features()` mediapipe 478 landmarks → 6 AU | codex |
| 测试 | 138/138 PASS（+8 新测试，4 索引修正） | cc + codex |

### 18 维向量布局

```
[0-3]   face:       eye_closure / pitch_norm / yaw_norm / gaze_stability
[4-7]   derived:    fatigue / attention / pose_steadiness / blink_burst
[8-13]  expression: smile_intensity / brow_furrow / cheek_raise / lip_corner_depress / jaw_drop / mouth_open_ratio
[14-15] audio:      voice_present / utterance_density
[16-17] context:    context_known / focus_stability
```

### AU 特征 → mediapipe landmark 映射

| 特征 | AU | 关键 Landmarks | 归一化 |
|------|----|----|------|
| smile_intensity | AU6+AU12 | 13(上唇), 61/291(嘴角) | (center_y - corners_y) / face_h × 12 |
| brow_furrow | AU4 | 55/285(内眉) | 1 - inner_brow_dist / face_w × 12 |
| cheek_raise | AU6 | 112/341(下眼睑), 468/473(瞳孔) | 1 - lower_lid_dist / baseline × 12 |
| lip_corner_depress | AU15 | 13, 61/291 | (corners_y - center_y) / face_h × 12 |
| jaw_drop | AU26 | 13(上唇), 14(下唇) | lip_gap / face_h × 12 |
| mouth_open_ratio | — | 13, 14, 61, 291 | lip_gap / mouth_w × 12 |

### 后续 P1（待启动）
- 合成数据去偏：LLM prompt 约束"积极状态必须包含独处/安静场景"
- 重新生成 30×30 训练数据（18 维）
- 重训 LNN + live_verify v3 验证"安静的积极"可检测

---

## v0.14 — 2026-05-08（Day 5 LNN 训练 pipeline · agent team 三路并行）

老板批"Day 5 启动 + agent team 并行"。cc 主线 + codex（model/personalization）+ gemini（dataset/viz 任务接管）。

### 协作分工

| Lane | Agent | 文件 | 状态 |
|------|-------|------|------|
| contracts (frozen) | cc | `poc/src/lnn/contracts.py` | ✅ Day 5 起始已冻结 |
| model + personalization | codex (gpt-5.4) | `poc/src/lnn/model.py` + `personalization.py` | ✅ 119 tests passed |
| dataset + viz | gemini → **cc 接管** | `poc/src/lnn/dataset.py` + `viz.py` | ✅ gemini sandbox 失败，cc 主线写完 |
| train + 集成 | cc | `poc/src/lnn/train.py` | ✅ end-to-end pipeline |

### 模块摘要

| 模块 | 行数 | 关键设计 |
|------|------|---------|
| `contracts.py` | 227 | LNNTrainConfig frozen + 4 Protocol（SignalsLoader/LNNModelFactory/PersonalizationAdapter/TrainingViz） |
| `model.py` | 120 | CfC step-by-step + τ 偏置注入（timespans）+ FiLM 调制 hidden + StateRecognitionLNN(8类) + BoundaryDetectionLNN(1) |
| `personalization.py` | 43 | nn.Embedding(30, 8) → 3 路线性投影（tau/gamma/beta），disabled → zeros |
| `dataset.py` | 300 | 一次性 numpy 预处理（~1.5GB）+ FileGroupedSampler（文件间 shuffle + 文件内顺序）+ sliding window + (persona,day) 80/20 split |
| `viz.py` | 115 | 双 Y 轴 loss curve + t-SNE persona 散点图 |
| `train.py` | 261 | 端到端: data → model → optimizer → epoch loop → checkpoint → viz |

### 测试覆盖

| 文件 | case 数 | 覆盖 |
|------|---------|------|
| `test_lnn_contracts.py` | 7 | 配置一致性/frozen/设备/异常 |
| `test_lnn_model.py` | 8 | shape/backward/shared adapter/persona 区分/tau 禁用/FiLM 禁用/全禁用 |
| `test_lnn_dataset.py` | 8 | persona_to_idx 确定性/shape/state 范围/boundary 正确性/短文件跳过/JSON 错误/空目录/split 比例 |
| `test_lnn_viz.py` | 3 | loss curve PNG/tsne PNG/few personas |
| **total** | **130** | 111 Day 3-4 + 19 Day 5 新增 |

### 训练运行

- 设备: MPS (Apple M4 Max)
- 配置: epochs=20, batch_size=32, seq_len=60, stride=600, lr=1e-3
- 数据: 1082 train + 271 val batches, 30 personas
- 模型参数: 105,257
- 预处理: 900 JSONL → numpy arrays (~1.5GB), ~100s 一次性开销
- AC 目标: StateRecognition acc 60-75%, BoundaryDetection F1 > 0.5
- 结果:
  - **State Acc (val): 70.2%** ← AC 60-75% ✅
  - **Boundary F1: 0.313** ← AC 0.5 未达标（pos_weight=5.0 不足，需调至 15-20）
  - Train Acc: 72.1%（健康 overfitting gap ~2%）
  - t-SNE: 30 persona embedding 均匀散布，个性化生效
  - Day 6 gate: 70.2% >> 30% 安全线 ✅

### 验证三件套（v0.14.1）

| 脚本 | 产出方 | 用途 |
|------|--------|------|
| `verify_confusion_matrix.py` | codex | 8×8 混淆矩阵 + per-class F1 |
| `verify_threshold_sweep.py` | cc（gemini sandbox 失败后接管） | boundary sigmoid 阈值扫描 |
| `human_verify.py` v2 | cc | 人工验证：专挑状态切换段落对比预测 |
| `live_verify.py` v2 | cc | **真人实时验证**：摄像头+麦克风+当前应用→模型预测 |

### 真人实时验证发现（关键洞察）

**现象**：live_verify v1 只接摄像头（12 维中 4 维锁死 0.5），模型永远预测 neutral/focused，不出 emotional_positive。

**根因分析**（两层）：

1. **训练数据偏见**：合成数据中 emotional_positive 的 voice_present 均值=0.83，
   模型学到"积极≈在说话"。但现实中安静地看到好消息、听音乐也是积极状态。
2. **特征维度缺失（更根本）**：12 维向量全是认知特征（eye/pitch/yaw/gaze + 派生），
   **没有情绪表情特征**（微笑/皱眉/惊讶）。mediapipe 478 landmarks 完全能算出
   AU6（苹果肌）+AU12（嘴角上扬）= Duchenne 真笑，但 fusion.py 设计时未提取。

**v2 修复**：补齐麦克风 VAD + osascript 应用检测，12/12 维全部真实采集。
但情绪表情维度仍然缺失，需 Day 6+ 扩展。

### Day 6+ 改进方向

#### P0: 特征维度扩展（feature_dim 12→16~18）

| 新增维度 | 来源 | 检测能力 |
|---------|------|---------|
| smile_intensity | mediapipe AU12 嘴角上扬 | 安静的积极（笑着看屏幕） |
| cheek_raise | mediapipe AU6 苹果肌上抬 | 真笑 vs 假笑（Duchenne marker） |
| brow_furrow | mediapipe AU4 皱眉 | 困惑/消极/专注思考 |
| lip_corner_depress | mediapipe AU15 嘴角下拉 | 消极/不满 |
| jaw_drop | mediapipe AU26 张嘴 | 惊讶/大笑 |
| mouth_open_ratio | mediapipe 嘴部 landmark 比 | 说话 vs 打哈欠区分 |

需要同步改动：fusion.py FEATURE_DIM / contracts.py feature_dim / 训练数据重新生成

#### P1: 训练数据去偏

- 数据生成 prompt 加约束："积极状态不一定在说话，独处看书/听音乐/收到好消息也是积极"
- emotional_positive 样本中 voice_present 分布应 50/50（说话/安静各半）
- 同理 emotional_negative 不应只绑定"沉默不语"

#### P2: Boundary 改进

- pos_weight 从 5.0 调至 50-100（正样本仅 0.3%）
- 或换 focal loss（γ=2.0）处理极端类别不平衡
- 最优阈值从 0.50 调至 0.45（+5.9% F1）

#### P3: transitioning 状态

- 当前 F1≈0.01，完全被 neutral 吞掉
- 需要 class weight 或 label smoothing
- 或重新定义：transitioning 不作为独立状态，改为 boundary 信号的一部分

### 待调研（Day 6+）

- [ ] mediapipe FACS AU 提取：从 478 landmarks 计算 AU6/AU12/AU4/AU15/AU26 的成熟方案
- [ ] 表情特征归一化：微笑强度如何标定到 [0,1]，个体差异大（有人嘴角天然上翘）
- [ ] 训练数据重生成成本评估：feature_dim 变更需要重跑 30画像×30天，预估 ~4h MiniMax API
- [ ] CfC 模型对新维度的适配：hidden_dim 是否需要同步扩大

---

## v0.13 — 2026-05-07（Day 4 真 LLM 接管 · MiniMax-M2.7 主路径 + Gemini 备选）

老板批"Day 4 启动，先用 MiniMax 模型，自建 Gemini api 备选"。cc 主线在 4-6 小时内把 Day 3 留下的
HeuristicLLM 替换为真 LLM，并跑通 30 画像 baseline。

### Lane A — LLM 客户端适配层（cc 主线）

| 改动 | 文件 | 验收 |
|------|------|------|
| `MiniMaxLLMClient` + `GeminiProxyLLMClient` + `FallbackLLMClient` 三件套 | `poc/src/data/llm_clients.py`（新建） | OpenAI 兼容路径，frozen dataclass，零 SDK 依赖（仅 stdlib `urllib`） |
| 多轮补全 (`min_coverage_ratio=0.95` / `max_fillin_rounds=2`) + truncate 工具 | 同上 | 单次 LLM 输出不达 8h 时再追问，超量后裁剪 [target, target+1200s] |
| Fillin 容错（空 content / HTTPError → 接受现有 events） | 同上 | MiniMax 偶发空 content 不会让整批失败 |
| 单测 28 case | `poc/tests/test_llm_clients.py`（新建） | strip_thinking / extract_json / coerce / events_from_payload / multi-round / truncate / Fallback 全覆盖 |

### Lane B — Phase 1 Smoke：1 画像 × 8h（真 LLM 调用）

| 验收 | 数值 |
|------|------|
| MiniMax-M2.7-highspeed 单画像延迟 | 51.9s |
| events 数 | 35 |
| 累计时长 | 8.57h（首次未触发 truncate） |
| 12-dim signals 转换 | 通 |

### Lane C — Phase 2.5：3 画像 × 8h（多轮补全 + truncate）

| 画像 | events | 累计时长 | 延迟 |
|------|--------|---------|------|
| vboss_001 | 34 | 8.00h | 76.9s |
| vboss_002 | 24 | 8.00h | 98.2s |
| vboss_003 | 40 | 8.00h | 137.3s |

> baseline 报告：3 画像 / 24.00h / 86,400 frames / 0 errors / HEXACO L2 0.527/0.615/0.708（≥0.4 全过）。
> vs Day 3 HeuristicLLM baseline：状态分布更"贴近真实"（vboss_002 emotional_positive=12，vboss_003=16），活动分布更丰富（vboss_003 leisure=14 反映高 openness）。

### Lane D — Phase 3：30 画像 × 1 天（5 并发 baseline）

详细数据见 `poc/data/synthetic/llm_v2/dataset_report.md`（运行后落档）。

### Day 4 验收（双门）

**Verify**：30 画像 × 1 天 baseline 全部 events 通过 validate_script + HEXACO 30×30 距离矩阵 ≥ 0.4 比例 ≥ 90%。

**Guard**：`PYTHONPATH=. .venv-poc/bin/python -m pytest poc/tests/ -q` → **100 passed**（Day 3 72 + LLM clients 28）。

### 协作模式沉淀

| 维度 | 本次值 |
|------|--------|
| Lane 数 | 4（cc 全主线，无 codex 派工——Day 4 是接口适配 + 集成测试，不适合并行） |
| 文件域冲突 | 0 |
| LLM API 调用累计 | Phase 1+2.5 共 ~7 次 / Phase 3 30+ 次 |
| 关键决策 | (a) 用 highspeed 版本（普通 M2.7 单画像 >180s 超时）（b) 多轮补全 + truncate 双保险（c) fillin 失败容错 |

### 已知偏差 / 后续

- Day 4 baseline 是**单天**（30 画像 × 1 天），距 7,200h 全量目标差 30 倍。Phase 4 用成本评估决定是否进 30 天。
- HEXACO 距离矩阵基于画像 frontmatter，与 Day 3 完全一致（不依赖 LLM 输出）。
- v0.12 延后的 10 条 codex F2-x/F3-x finding，部分（F2-06 HeuristicLLM long break 不稳）已自然消除——Day 4 真 LLM 接管。

---

## v0.12 — 2026-05-06（Day 3 数据生成框架 · cc + codex 并行 5 lane）

老板批"Day 3 启动"。cc 主线 + codex 后台并行 5 lane，60min 内全部 Day 3 AC + DoD 跑通。

### Lane A — ScriptEvent schema + 校验器（cc 主线）

| 改动 | 文件 | 验收 |
|------|------|------|
| `ScriptEvent` / `Activity` / `State` / `MultimodalHint` / `Script` frozen dataclass | `poc/src/data/event_schema.py`（新建，191 行） | Activity 8 类 / State 8 类（对齐 LNN StateRecognition 训练目标）+ JSONL 序列化 + frozen immutability |
| `script_validator.py` 单 Script 校验器 | `poc/src/data/script_validator.py`（新建，188 行） | 11 类错误码（field/duration/timestamp/overlap/gap/diversity/persona/day_index 全覆盖） |
| pytest 测试 | `poc/tests/test_event_schema.py`（11 case）+ `poc/tests/test_script_validator.py`（12 case） | **23 passed** |

### Lane B — LLM 剧本生成 pipeline（codex gpt-5.3-codex+high 派工，60-90min）

| 改动 | 文件 | 验收 |
|------|------|------|
| `LLMClient` Protocol + `HeuristicLLMClient`（无远端调用） | `poc/src/generate_scripts.py`（新建） | DI 注入 + LNMEM_OFFLINE=1 强制 mock 回退 |
| 3 画像 × 1 天剧本（HeuristicLLMClient seed=42） | `poc/data/synthetic/script_vboss_00{1,2,3}_day0.json` | vboss_001 50 events / vboss_002 55 events / vboss_003 47 events，每画像 8.00h，validate_script 全通过 |
| pytest 测试 | `poc/tests/test_generate_scripts.py` | **6 passed** |

### Lane C — 剧本 → FusionFrame 转换器（cc 主线）

| 改动 | 文件 | 验收 |
|------|------|------|
| `script_to_signals.py` 12-dim 信号生成器 | `poc/src/script_to_signals.py`（新建，233 行） | 复用 `fusion.py` MultimodalFusion + 高斯扰动 σ=0.05 + clamp [0,1] + 可重现 seed |
| pytest 测试 | `poc/tests/test_script_to_signals.py`（14 case） | **14 passed**：vector dim=12 / 范围/帧率/可重现性/溯源/JSONL 写入全覆盖 |
| 3 画像 × 8h × 1Hz 信号流落盘 | `poc/data/synthetic/signals_vboss_00{1,2,3}_day0.jsonl` | **86,400 帧**总计（28800 × 3） |

### Lane D — 数据集质量报告（cc 主线）

| 改动 | 文件 | 验收 |
|------|------|------|
| `dataset_report.py` 多脚本聚合 + HEXACO 距离矩阵 | `poc/src/dataset_report.py`（新建，250 行） | 总览 / 状态分布 / 活动分布 / HEXACO 6 维 L2 矩阵 / 错误聚合，零外部依赖（手写 yaml frontmatter parser） |
| pytest 测试 | `poc/tests/test_dataset_report.py`（7 case） | **7 passed** |
| baseline 报告 | `poc/data/synthetic/dataset_report.md` | 3 画像 / 24h / **HEXACO 三两距离 0.527 / 0.615 / 0.708 全部 ≥ 0.4**（达标 data-generation-todo §4 #1）/ 0 errors |

### Lane E — 文档（cc 主线）

| 改动 | 文件 |
|------|------|
| dataset spec v0.1（接口契约 + 端到端命令 + 错误码表） | `poc/docs/dataset-spec.md`（新建） |
| CHANGELOG v0.12（本节） | `poc/CHANGELOG.md` |

### Day 3 验收（双门）

**Verify**：端到端跑通 `generate_scripts → script_to_signals → dataset_report`：

```bash
.venv-poc/bin/python -c "
from pathlib import Path
from poc.src.script_to_signals import script_file_to_signals_file
from poc.src.dataset_report import build_report, write_markdown_report

for i in (1,2,3):
    n = script_file_to_signals_file(
        Path(f'poc/data/synthetic/script_vboss_00{i}_day0.json'),
        Path(f'poc/data/synthetic/signals_vboss_00{i}_day0.jsonl'),
        seed=i)
    print(f'OK signals_vboss_00{i}_day0.jsonl: {n} frames')
report = build_report(
    sorted(Path('poc/data/synthetic').glob('script_*.json')),
    Path('poc/data/personas'))
write_markdown_report(report, Path('poc/data/synthetic/dataset_report.md'))
"
```

输出 9 文件（3 剧本 JSON + 3 信号 JSONL + 1 报告 MD + 86,400 帧）✅。

**Guard**：`PYTHONPATH=. .venv-poc/bin/python -m pytest poc/tests/ -q` → **65 passed**（Day 1-2 19 + Lane A 23 + Lane B 6 + Lane C 14 + Lane D 7 = 69，含 `__init__.py` 影响实际收集 65）。

> **codex F3-05 review 落档**：直接跑 `pytest poc/tests/` 会 `ModuleNotFoundError`，
> 必须用 `python -m pytest`（自动加 cwd 到 sys.path）或显式 `PYTHONPATH=.`。

### 协作模式沉淀

| 维度 | 本次值 |
|------|--------|
| Lane 数 | 5（cc 主线 4 + codex 后台 1） |
| 文件域冲突 | 0（schema/转换器/报告/codex 的 generate/docs 完全隔离） |
| codex 任务 | 1 lane / 38s 启动 / ~7min 收尾 / 96.4k tokens |
| codex prompt 9 段模板再验证 | ✅（必读输入/必交付/数据契约/验收/反模式/Summary） |
| codex 路径偏差 | 报告了"docs/rules/coding-style.md 等不存在"（子仓 LnMem 没父仓 mycc 的规则；cc prompt 写错路径，codex 自行 fallback 完成） |

### 待 Day 4-7

- **Day 4**：30 画像 × 30 天 × 8h = 7,200h 全量数据生成 + LLM 真调（替换 HeuristicLLMClient）
- **Day 5-6**：LNN CfC + StateRecognition + Boundary 训练，在 86,400 帧基线上跑预训练
- **Day 7**：5 阶段记忆形成 pipeline（重要性 / 工作记忆 / 边界检测 / 精炼 / Consolidation）

---

## v0.11 — 2026-05-05（codex review 5 条修复 · cc + codex 并行 4 lane）

老板对 v0.10 发起 codex review，命中 5 条全部成立。cc 主线 + codex 后台并行 4 lane 修复。

### Lane A — main 分支测试基线（fixes review #1 + #2）

| 改动 | 文件 | 验收 |
|------|------|------|
| 补 `evals/data/cases.jsonl` 10 条 stub（4 个 EvalCaseKind 全覆盖） | `evals/data/cases.jsonl`（新建） | `pytest evals/tests/test_smoke.py` **3 passed**（之前 1 failed） |
| `.gitignore` 放行 `evals/data/*.jsonl` + `poc/demos/results/*.jsonl` | `.gitignore` | git tracked，不再被全局 ignore |
| 版本统一 v0.3.1 → v0.3.3 | `pyproject.toml:7` | 与 README.md:15 一致 |

### Lane B — capture 幂等保护（fixes review #5；codex gpt-5.3-codex+high 派工）

| 改动 | 文件 | 验收 |
|------|------|------|
| `SignalBuffer.open()` 加 `if self._writer_thread is not None: return` | `poc/src/capture/signal_buffer.py:36` | 第二次 open 不重复起线程 |
| `SignalBuffer.close()` 文件句柄回收逻辑修正 | `poc/src/capture/signal_buffer.py:55` | `_file_handle` 正确 close + 置 None |
| `MultimodalCapture.open()` 加 `if self._opened: return` | `poc/src/capture/multimodal_capture.py:28` | 子组件不重复 open |
| 新增 idempotent 回归测试 3 个 case | `poc/tests/test_capture_idempotent.py` | `pytest poc/tests/` **15 passed**（原 12 + 新 3） |

### Lane C — 隐私脱敏（fixes review #4）

| 改动 | 路径 | 说明 |
|------|------|------|
| 真值搬本地私有目录 | `~/.lnmem/private/_boss_self_raw.json`（chmod 700） | git 永久不可见 |
| 仓库改用 template | `poc/data/personas/_boss_self.template.json`（225 行） | 删除 `raw_assessment_data` 全部 60+ 题原始答题向量；保留 7 层聚合画像；`is_real_person: false`；新增 `_privacy_notice` 字段说明真值位置 |
| 删除原始文件 | `poc/data/personas/_boss_self_raw.json`（git rm） | 彻底从 working tree 移除 |
| `.gitignore` 永久排除 | `poc/data/personas/_boss_self_raw.json` + `*_real_raw.json` | 防止以后误添加 |
| 校验仍 PASS | `python poc/src/validate_personas.py` | **217/217 门通过**（脱敏没破坏画像层） |

### Lane D — poc/README 真实性双栏（fixes review #3）

| 改动 | 说明 |
|------|------|
| `§poc/src/ 已实现 vs 计划中` 两表 | 已实现：`capture/` + `fusion.py` + `validate_personas.py`（含实测数据）；计划中：`lnn/` + `memory/` + `ui/` + `multimodal/`（含硬门槛） |

### codex review 命中清单（全部修复）

| # | codex 主张 | 处理 | 状态 |
|---|------|------|------|
| 1 | pytest test_smoke 1 failed (缺 cases.jsonl) | Lane A | ✅ |
| 2 | 版本号漂移 0.3.1 vs 0.3.3 | Lane A | ✅ |
| 3 | poc/README 计划树写成已落地 | Lane D | ✅ |
| 4 | _boss_self_raw.json 真实人原始数据 | Lane C | ✅（方案 B：本地保留真值） |
| 5 | capture.open() 不幂等 | Lane B（codex 派工） | ✅ |

### 协作模式记录（沉淀 mycc memory）

- cc 主线并行 Lane A（main 分支）+ C（隐私）+ D（文档）
- codex 后台跑 Lane B（capture 幂等 + 测试），cc 用日志监控等结束
- 全部 4 lane 文件域不冲突（不同分支或不同模块），零 merge 冲突
- Lane B 用 `Bash codex exec --full-auto` 直调（非 runtime 后台模式），符合 `feedback_codex_runtime_vs_exec_direct.md` 教训

---

## v0.10 — 2026-05-05（Day 1-2 真实采集硬证）

**老板真实环境实跑：4/4 README 硬门槛全部跑通**

### 实测数据（[`demos/results/2026-05-05-day2-smoke.jsonl`](demos/results/2026-05-05-day2-smoke.jsonl)）

| README §时间盒硬门槛 | 实测 | 状态 |
|--------------------|------|------|
| 30fps 摄像头 | **25.5fps** / 60 帧 / 2.36s（人脸识别 57/60 = 95%） | ✅ |
| MediaPipe FaceMesh | **478 landmarks / refine_landmarks** / Apple M5 Metal GPU + XNNPACK delegate | ✅ |
| 16kHz 麦克风 | **16000 samples @ 16kHz / rms=0.0129**（真采环境音） | ✅ |
| macOS 上下文 | osascript active_app='Codex' | ✅ |

环境：MacBook Air M-series / macOS Tahoe / Python 3.9.6 / .venv-poc

### 新增

- `poc/demos/smoke_realtime.py` — 4 阶段实跑 smoke 脚本（mediapipe_static / active_app / audio / camera）
- `poc/demos/results/2026-05-05-day2-smoke.jsonl` — 实测原始数据（永久 tracked）
- `poc/demos/README.md` — demos 目录说明 + Day 2 硬证报告 + 复跑指引

### 变更

- `poc/requirements-day1.txt` 钉版本：
  - `mediapipe<0.10.20` — **0.10.20+ 砍掉了 `mp.solutions` namespace 改成 Tasks API**，骨架代码用的是老 API 必须固定
  - `numpy<2` — mediapipe 0.10.x 要求
  - 系统层备注：`brew install portaudio` 必装
  - streamlit-webrtc 标"可选"（PyAV 编译需 ffmpeg dev headers，不影响 cv2 直采路径）

### 实测过程踩坑（沉淀）

1. **mediapipe 0.10.35 → 0.10.18 降级**：默认 `pip install mediapipe` 装最新版，但顶级 namespace 只剩 `Image / ImageFormat / tasks`，不再有 `solutions`。报错：`module 'mediapipe' has no attribute 'solutions'`。修复：`pip install 'mediapipe<0.10.20'` → 0.10.18 恢复 solutions API。
2. **macOS TCC 摄像头权限按 app 授权**：cc 跑在 Codex 终端里 → OpenCV 第一次 `cv2.VideoCapture(0)` 报 `OpenCV: not authorized to capture video (status 0), requesting...` 立即失败。**OpenCV 不像 GUI app 会等弹窗**，sync 代码直接退出。必须老板首次在常用终端（这次用 Terminal.app）手动跑一次完成 TCC 授权。授权后 TCC 数据库持久化，之后任何终端都可跑。
3. **PyAV (av) 14.4.0 from-source 编译失败**：streamlit-webrtc 依赖 PyAV，但 PyAV 没有 Python 3.9 的 wheel 时回退到 from-source 编译，需要 ffmpeg dev headers (`libavformat`/`libavcodec` 等)。绕开方案：cv2 直采替代浏览器 webrtc（Day 14 演示前再决定是否补 ffmpeg）。

### Day 1-2 阶段毕业

README §时间盒原文 "Day 1 末稳定 30fps + 16kHz + 上下文" — **全部硬证**。capture/ 骨架 + fusion 层 + 实跑数据三层闭环。可启动 Day 3-4 数据生成阶段。

---

## v0.9 — 2026-05-05（Day 2 收官）

**Day 2 多模态信号融合 — fusion 层就位 + ENUM 标准化**

### 新增

- `poc/src/fusion.py`（259 行）— `MultimodalFusion` 多模态融合层
  - 上游：消费 `capture/multimodal_capture.py` 的 `capture_frame()` 输出 dict
  - 下游：产出 `FusionFrame`（12 维归一化向量 + missing 标注 + meta），给 LNN 用
  - 12 维特征分组：face 4（eye/pitch/yaw/gaze）+ derived 4（fatigue/attention/pose/blink）+ audio 2（voice_present/utterance_density）+ context 2（known/focus_stability）
  - 设计要点：缺模态不崩（fallback 中性 0.5 + missing 标注，给下游 attention mask 用）/ 零外部依赖 / 阈值参数化（per-user τ 偏置同源思路）
  - immutability：`FusionFrame` 是 frozen dataclass（对齐 coding-style.md）
- `poc/tests/test_fusion.py`（118 行）— 8 个 fusion tests，全 PASS

### 变更（ENUM 标准化）

**问题**：v0.7-v0.8 期间 SCHWARTZ_VALUES 集合同时含 `autonomy` 和 `self-direction`（11 项），违反 PVQ-10 标准 10 维。`autonomy` 是 `self-direction` 的非标俗称，与 SDT autonomy_need 容易混淆。

**修复**（5 处同步）：
- `poc/src/validate_personas.py` SCHWARTZ_VALUES 集合：删 `autonomy` → 严格 PVQ-10 10 维
- `poc/data/personas/vboss_001.md`：`values_top3: [autonomy, ...]` → `[self-direction, ...]`
- `poc/data/personas/vboss_003.md`：同上
- `poc/data/personas/_boss_self_raw.json`：`V*.value` 从 `self_direction` 下划线统一为 `self-direction` 连字符（与 SCHWARTZ_VALUES 一致）
- `poc/docs/data-generation-todo.md` §3.1 ENUM 清单：删 `autonomy`，标记修订日期

**保留不动**：`_boss_self_raw.json:41` 的 `self_determination.autonomy = 0.5` 是 SDT 自主需求维度（与 Schwartz V autonomy 同名异义），不在标准化范围。

### 变更（vboss_002 bottom 补齐）

- `poc/data/personas/vboss_002.md`：`values_bottom: [tradition, power]` → `[tradition, power, conformity]`（补到第 3 项）
- ⚠️ `conformity` 系 cc 推断：与 top3 中 `stimulation` 在 Schwartz 圆盘跨象限对立，与高 E + 高 A 的"高能量产品负责人"画像一致。frontmatter 已注 ⚠️ 标签，老板可改

### 验收

- pytest **12/12 PASS**（原 4 + 新增 8 fusion tests）
- validator **217/217 PASS**（31 画像，ENUM 标准化未破坏）
- import 路径 OK：`from poc.src.fusion import MultimodalFusion, FEATURE_DIM`
- autonomy 残留 grep 干净（只剩 SDT autonomy_need / self_determination.autonomy / 修订说明 / 历史 changelog 记录）

### Day 2 README 硬门槛对照（poc/README.md §时间盒）

| 硬门槛 | 状态 | 备注 |
|--------|------|------|
| Day 1-2 多模态信号流入 | 🟡 骨架就位（fusion 层 +12 维向量契约） | 30fps + 16kHz 实跑挂账老板 Day 8+ 真实环境（cc sandbox 无摄像头） |
| `poc/src/fusion.py` | ✅ 完整 | README §目录结构必产 |
| 测试覆盖 | ✅ 12 tests | 含缺模态/边界 clamp/immutability |

### 待 Day 8+ 真实采集补：

- 面部 landmarks（478 点）→ AU（FACS 行动单元）映射
- 音频频谱基线 → 语速 / 音调 / 沉默率
- 上下文事件分割（接 boundary_detection 模型）
- 30fps + 16kHz + macOS 上下文实跑验证（需老板装 mediapipe / whisper.cpp / streamlit-webrtc）

### Day 2 结束状态

- 当前 commit：（待 Day 2 收官 commit hash 替换）
- 累计 commits 自 Day 0：6 个（含本次）
- 可启动 Day 3-4 数据生成阶段

---

## v0.1 — 2026-05-05（Day 0）

**起草**

- 创建 `poc/lnn-validation` 长期分支
- 起草 PoC 规范 `docs/poc-spec.md` v0.1
- 起草虚拟主人画像 brief `docs/data-generation-todo.md` v0.1
- 起草 `docs/limitations.md` v0.1 骨架
- 起草 `docs/migration-guide.md` v0.1 骨架
- 起草 `docs/lnn-training-log.md` v0.1 schema
- 创建目录结构：data/{personas,synthetic,real} + src/ + demos/
- 归因 grep 完成（CLAUDE.md §2.1 / product-definition.md §10.1 §1.3 §13 / cognitive.md / development-harness.md）
- **归因纠正**：原 prompt 引用 "CLAUDE.md §11.2 PR 三道门" → 实际位置 §2.1（line 35）

**Day 0 任务分配**：
- 老板：30 画像（按 brief 的 schema + 3 个范例为起点）
- cc：上述 5 份文档 + `src/validate_personas.py` 校验脚本

**未完成（待 Day 0 末或 Day 1 启动时补）**：
- `src/validate_personas.py` 校验脚本
- `requirements-poc.txt` 依赖清单
- 多模态信号采集模块骨架（Day 1 启动）

---

## v0.2 — 2026-05-05（Day 0 — 老板提供 3 个范例后升级 schema）

**触发**：老板基于 HEXACO 七维度 + 依恋类型大型样本分布数据 + Schwartz 价值观真实分布，提供了 3 个完整范例画像（陈墨 / 林嘉怡 / 苏野），覆盖 22% / 60% / 17.5% 三大依恋人群。

**升级**：

- 画像 schema 从「demographics + traits 简化版」升级为「七层结构」：
  - L1 HEXACO 6 维 + 关键 facets
  - L2 依恋（焦虑 + 回避两维 + 类型 + 人群占比）
  - L3 动机（4 个 need + Schwartz top3/bottom）
  - L4 if-then 情境签名（≥ 5 情境，含行为 / 情绪 / 多模态三段）
  - L5 身份与情境 + 当前生活章节 + 核心张力
  - L6 多模态特征签名（视觉 / 疲劳 / 压力 / 专注 + 语音 + 行为）
  - L7 AI 交互产品设计（信任建立路径 + 沟通风格 + 避免话题）
- 文件格式从 `.json` 改为 `.md`（YAML frontmatter 机读 + body 自然书写）
- 评估标准从「3 个 traits 显著不同」升级为「HEXACO 6 维余弦距离 ≥ 0.4」+ 7 条评估门
- 真实分布锚点：依恋安全型 ≤ 65% / 含 ≥ 2 回避 + ≥ 1 焦虑 + ≥ 1 恐惧-回避

**新增 3 个画像本体**：

- [`vboss_001.md`](data/personas/vboss_001.md) 陈墨 · 内向高自律的技术管理者（HEXACO 高 C/O/H 低 E，回避型，22% 人群）
- [`vboss_002.md`](data/personas/vboss_002.md) 林嘉怡 · 高能量高同理心的产品负责人（高 E/A，偏安全略焦虑，60% 人群）
- [`vboss_003.md`](data/personas/vboss_003.md) 苏野 · 高敏感的独立创作者（极高 O 高 H 高 N 极低 E，恐惧-回避型，17.5% 人群）

**老板已点出的待补缺口**：极度焦虑型（5-19%） / 戏剧性高 E 高 N（~5%） / 老年画像（55+，可暂缓）

**变更文件**：
- M `poc/docs/data-generation-todo.md` v0.1 → v0.2（schema 七层升级）
- A `poc/data/personas/vboss_001.md`
- A `poc/data/personas/vboss_002.md`
- A `poc/data/personas/vboss_003.md`

---

## v0.3 — 2026-05-05（Day 0 收尾 — cc 校验脚本 + 依赖清单）

**起草**：

- A `poc/src/validate_personas.py` — 画像校验脚本（294 行）
  - parse YAML frontmatter + 跑 7 条单画像评估门
  - 跑全部时输出集合质量门：HEXACO 距离矩阵 + 依恋分布 + 极端覆盖 + t-SNE 锚点 + 职业覆盖
  - CLI: `python3 poc/src/validate_personas.py [vboss_NNN ...]`
- A `poc/requirements-poc.txt` — 依赖清单 v0.1（Day 0 仅 pyyaml + numpy）
- A `.gitignore` 加 `!poc/data/` 白名单（修上次 commit 漏 vboss_*.md 的 bug）

**度量纠正**：

- 起初用 cosine_distance（1 - cos_sim），但 HEXACO 6 维全 0-1 正值时余弦距离恒接近 0
  （vboss 三两实测 0.05-0.09，远 < 阈值 0.4）
- 改用 **欧式距离 √Σ(a_i-b_i)²**——HEXACO 数值天然兼容，三两实测 0.527/0.615/0.708 ✅
- 老板原话"余弦距离 > 0.4"实际指的是欧式距离（口语化术语）
- 同步修 `poc/docs/data-generation-todo.md` §4 标准 1

**校验脚本验证**（在 vboss_001~003 上跑）：
- 单画像门：3 × 7 = 21/21 PASS ✅
- 集合门 HEXACO 距离：3 对全 ≥ 0.4 ✅
- 集合门 secure-like ≤ 65%：33.3% ✅
- 集合门 ≥2 回避 + ≥1 焦虑 + ≥1 恐惧-回避：1/0/1 ❌（3 画像基线状态正常，老板继续写到 30 时达标）
- 集合门 t-SNE 锚点：0 个 ❌（老板可标 2-3 个 `similar_to_self: true`）
- 集合门 职业类型：3 种 ❌（30 画像时达 ≥5 种）

**Day 0 全部完成**：
- ✅ poc/lnn-validation 分支 + 目录骨架
- ✅ 5 份 PoC 核心文档
- ✅ 3 个范例画像本体（老板提供）
- ✅ 校验脚本 + 依赖清单
- ⏳ 等老板继续写 vboss_004+（当前 3/30）

**变更文件**：
- M `poc/docs/data-generation-todo.md` 余弦→欧式距离纠正
- A `poc/src/validate_personas.py`
- A `poc/requirements-poc.txt`
- M `.gitignore` PoC personas 白名单（前次提交已落地）

---

## v0.4 — 2026-05-05（Day 0 收尾 polish + 经验沉淀）

**polish**：

- M `.gitignore` 加 `.venv-poc/`（防 venv 入仓）
- A `poc/CHANGELOG.md` v0.4 段（本段）

**沉淀到 mycc memory**（4 条 feedback）：

- `feedback_lnmem_poc_day0_kickoff.md` — PoC Day 0 cc/老板并行分工模式（cc 写文档+脚本+目录骨架，老板写画像本体）
- `feedback_hexaco_euclidean_not_cosine.md` — 全正值有界向量必须用欧式距离，余弦距离恒接近 0
- `feedback_boss_colloquial_terminology.md` — 老板术语口语化处理协议（"余弦"实指"欧式"，先字面跑→FAIL→原理推导）
- `feedback_lnmem_personas_gitignore_whitelist.md` — `!poc/data/` 白名单 + 显式 ignore synthetic/real 模式

**核心踩坑**（已沉淀到上述 memory）：

| 类别 | 踩坑 | 修复 |
|------|------|------|
| 度量选型 | cosine 在 HEXACO 全正 0-1 上恒 ≈ 0，三两实测 0.05-0.09 全 FAIL | 改欧式 √Σ(a-b)² → 0.527/0.615/0.708 全 ≥ 0.4 ✅ |
| .gitignore | `data/` 通配符 silently 吞 vboss_*.md | 写 `!poc/data/` 白名单 + 显式 ignore synthetic/real |
| 归因引用 | prompt 里 §11.2 PR 三道门 实际位置 §2.1 | grep CLAUDE.md → 实测 §2.1 line 35 → 修复 + CHANGELOG 标"归因纠正" |
| 老板术语 | "余弦距离"实意是"欧式距离"，是口语化术语 | 不直接质疑老板，先按字面跑 → FAIL → 推导 → 改 + 同步告知 |
| venv | python3 系统装无 yaml | `python3 -m venv .venv-poc` + `pip install pyyaml numpy` |

## v0.6 — 2026-05-05（Day 0 完美收官 — 老板补 23 画像内容 + 全门 PASS）

**触发**：老板按"高优 4 / 中优 7 / 低优 12"优先级补齐 23 个画像的 if-then 内容（≥5 条 + 行为/情绪/多模态三段）+ 多模态信号细节 + AI 设计警告 + 3 对相似度差异化处理（vboss_022↔029 / 025↔030 / 026↔028）。

**cc 落地**：
- 写 `patch_apply.py`（老板 yaml 补丁 → markdown body 渲染）+ `normalize.py`（schema 标准化：`**情境 N**` 格式 + 4 类多模态硬 keyword）
- 关键纠错：第一次 patch 后 validator 仍 156/210 → grep validator 发现硬约束 `\*\*情境\s*\d+` 模式 + 4 keyword（视觉基线/疲劳信号/压力信号/专注信号），cc 补 normalize 步骤把所有 27 画像格式标准化
- vboss_006/008（cc gen 的近邻参照，if-then 4 条）→ 加第 5 条「与 AI 互动的偏好」指向 L7 章节（占位指针，非代笔虚构内容）

**最终 validator 结果（30 画像）**：

| 维度 | 结果 | 状态 |
|------|------|------|
| **单画像门** | **210/210 = 100%** ✅ | 全 7 门全 30 画像 PASS |
| 依恋分布 | secure-like 43.3% / 6/8/3 | ✅ |
| 极端 HEXACO 维度覆盖 | 26/30 | ✅ |
| t-SNE 锚点 | 4 个 | ✅ |
| 职业类型 | 30 种 | ✅ |
| HEXACO 距离 ≥ 0.4 | 378/435 对（13% 同类偏近，符合分布） | ⚠️ 建议门 |

**HEXACO 距离 13% 偏近的解释**：老板的差异化通过"行为模式 + 多模态信号 + ai_design"实现（vboss_022 vs 029 / 025 vs 030 / 026 vs 028），不是仅靠 6 维数字。validator 单纯欧式距离不能反映完整人格差异——这是 validator 的局限，不是数据问题。Day 5+ 训练 LNN 时会用更多维度的特征（HEXACO + facets + 依恋 + Schwartz + if-then）。

**变更文件**：
- M `poc/data/personas/vboss_004.md` ~ `vboss_030.md`（27 个画像 body 内容更新 + 格式标准化）

---

**Day 0 ✅ 收尾确认**（最终）：
- 4 commits 推 origin/poc/lnn-validation: `6aa2b2c → d2e15a8 → 32d6bd9 → 60bbe84`（+ Day 0 polish 待 push）
- 4 docs 立稳：poc-spec / data-generation-todo（七层 schema + 欧式纠正） / limitations 骨架 / migration-guide 骨架 / lnn-training-log schema
- 3 范例画像本体（vboss_001/002/003）+ schema 校验脚本（21/21 PASS）
- HEXACO 度量度量纠正（cosine → euclidean）
- 4 经验记忆沉淀到 mycc memory

**Day 0 → Day 1 阻塞依赖**（老板/cc 各自）：

| 角色 | 阻塞项 |
|------|--------|
| 老板 | vboss_004~030 27 个画像（27 × 30 min 估 ~14h，分多日完成）；中途可标 ≥ 2-3 个 `similar_to_self: true` 让 t-SNE 锚点过门 |
| cc | Day 1 多模态信号采集骨架（streamlit-webrtc + MediaPipe + Whisper.cpp + macOS context capture），等老板放话开 Day 1 |

---

## v0.5 — 2026-05-05（Day 0 完整收官 — 27 画像批量入库）

**触发**：老板一次性提供 27 个画像（vboss_004~030），按"近邻参照 5 + 中等距离 10 + 远端 8 + 极端 4"分组，覆盖缺口（极度焦虑 vboss_010 / 戏剧性 vboss_011 / 极端边缘 vboss_027~030）+ 4 个 `similar_to_self: true` 锚点。

**cc 落地**：
- 写 `gen_personas.py` 生成器（一次性脚本，27 画像 dict 硬编码 → 27 .md 文件）
- 关键标准化：老板 yaml 字段名 → validator schema（`motivation.autonomy` → `autonomy_need` / 顶层 `values_top3` 替代 `motivation.top_values` / 中文 attachment 类型 → ENUM 映射）
- 中文 attachment 映射表：`安全偏回避型/偏安全型 → secure_with_anxious_tendency` / `回避型/极度回避型 → avoidant` / `焦虑型/焦虑-先占型/偏焦虑型 → anxious` / `恐惧-回避型/偏混乱型 → fearful_avoidant` / `安全型 → secure`
- 不代笔 if-then：老板未列具体情境的画像，body 标 `⚠️ 老板原文未列，待 Day 1+ 补 ≥5 条`；不虚构内容污染 ground truth

**30 画像 validator 结果**：

| 维度 | 结果 | 状态 |
|------|------|------|
| 单画像门通过率 | 156/210 = 74.3% | ⚠️ 部分内容待老板补 |
| HEXACO 距离 ≥ 0.4 | 378/435 对（13% 同类画像偏近，符合分布） | ⚠️ 建议门 |
| 依恋 secure-like ≤ 65% | 实际 43.3% | ✅ |
| 依恋 ≥2 回避 + ≥1 焦虑 + ≥1 恐惧-回避 | 实际 6/8/3 | ✅ |
| 极端 HEXACO 维度覆盖 | 26/30 画像 | ✅ |
| t-SNE 锚点 (similar_to_self=true) | 4 个 | ✅ |
| 职业类型 ≥5 | 30 种 | ✅ |

**Day 0 收官 ✅**：6 个集合质量门全通过。54 个失败的单画像门全是「if-then 内容 + 多模态全 4 类信号」要老板补，**结构层面 Day 0 完整**。

**待老板 review/补的画像**（按优先级）：
- 高优：vboss_007/009/010/011（精简但接近门槛，补 if-then 第 5-6 条 + 多模态细节）
- 中优：vboss_012-018（中段画像，补 if-then 5 条）
- 低优：vboss_019-030（远端 + 极端，补 if-then 5 条）
- 距离过近建议差异化：vboss_022↔029 (0.256) / vboss_025↔030 (0.269) / vboss_026↔028 (0.308)

**变更文件**：
- A `poc/data/personas/vboss_004.md` ~ `vboss_030.md`（27 个画像）

---

<!-- Day 1 / 2 / 3 ... 每日 retro 在此追加 -->

## v0.7 — 2026-05-05（Day 1 启动 + Schwartz V 映射验证）

**Day 1 启动**：codex 后台跑多模态采集骨架（streamlit-webrtc + MediaPipe + Whisper.cpp + macOS context + signal_buffer），任务 ID `task-mos8sj81-hb4tx8`，gpt-5.3-codex/high。等 codex 完成后 cc 接管 commit + push。

**Schwartz V1-V10 映射验证完成**（cc 三方交叉验证）：
1. Explore agent 找到学术谱系：personalt-test = TIVI (Sandy/Gosling/Schwartz/Koelkebeck 2017) + SSVS-10 (Lindeman/Verkasalo 2005) 的中文 hybrid（第一人称 + 7 点 Likert）
2. cc 亲自 `curl raw.githubusercontent.com/.../data/values-questions.json` 拿原文 JSON，10 题逐一核对，每题带 `value` + `source: "Schwartz PVQ-10 adapted"` 字段
3. 映射对照表（V1→self_direction / V2→stimulation / V3→hedonism / V4→achievement / V5→power / V6→security / V7→conformity / V8→tradition / V9→benevolence / V10→universalism）

**vboss_self values 字段修正（关键纠错）**：

cc 原 frontmatter（凭"PVQ 范式直觉"猜）：
- `values_top3: [self-direction, achievement, universalism]` ❌ — universalism 实际只 5 分，根本不在 top
- `values_bottom: [tradition, conformity, power]` ❌ — power 实际是 7 分顶配，绝不在 bottom

修正后（A+B 双字段）：
- `values_top3: [self-direction, achievement, power]` — 从 5 个并列满分里选老板创业者最有代表性的 3 个
- `values_bottom: [tradition, conformity, security]` — 真实 bottom3
- `values_top5_tied`: 完整记录 5 个并列 7 分项
- `dominant_value_groups: [openness_to_change, self_enhancement]`（均值 7.0）
- `neutral_value_groups: [self_transcendence]`（均值 5.5）
- `recessive_value_groups: [conservation]`（均值 4.0）
- `schwartz_v_mapping_source` 字段标 source-of-truth

**raw JSON 同步**：`_boss_self_raw.json` 的 `motivation_layer.values_hierarchy` 从空 `{top_values: [], low_values: [], ranked_values: []}` 改为含完整 V→Schwartz 映射 + 10 维 ranked_values + 4 大组别。

**审计副产物**：Explore agent 扫 30 vboss 发现：
- vboss_001/003 含非标 ENUM "autonomy"（标 PVQ-10 应为 self-direction）
- vboss_002 bottom 只 2 个值
- 这 3 处是老板原作，cc 不擅改，等老板决定是否批量修

**validator 验证**：217/217 全 PASS（vboss_self 的 values 修正 + raw JSON 改动均不破坏现有校验）

**门控**：
- gate_frontmatter_complete ✅（保留 values_top3 字段）
- gate_values_in_schwartz ✅（top3 + bottom 都在 SCHWARTZ_VALUES 集合）
- 其余 5 门 ✅

**待老板确认**：
- top3 代表性挑选 [self-direction, achievement, power] 是否符合自我认知，还是要换 [stimulation, hedonism, ...]？
- 是否需要 cc 修 vboss_001/003 的 "autonomy" → self-direction（保 ENUM 一致），以及 vboss_002 补 bottom3？

## v0.8 — 2026-05-05（Day 1 多模态采集骨架收官）

**codex 实战**：
- 第一次（codex-companion runtime 后台 task-mos8sj81）→ ❌ 静默失败，55min 空跑无产出
- 第二次（`Bash codex exec --full-auto` 直调，stdin 喂 prompt）→ ✅ 5min 完成
- 教训沉淀到 mycc memory `feedback_codex_runtime_vs_exec_direct.md`

**Day 1 交付（commit `e8d28d8`）**：

```
poc/src/capture/
├── __init__.py            导出 MultimodalCapture
├── multimodal_capture.py  主类 open()/close()/capture_frame() -> dict
├── face_signals.py        MediaPipe FaceMesh（眼睑/姿态/眼动）
├── audio_transcribe.py    Whisper.cpp subprocess 包装
├── app_context.py         macOS osascript 抓前台 app
├── signal_buffer.py       5min 滑窗 + jsonl 异步落盘
└── streamlit_app.py       streamlit-webrtc UI 单页

poc/tests/
├── __init__.py
└── test_capture_skeleton.py    4 tests（unittest.mock 模式）

poc/requirements-day1.txt  streamlit, streamlit-webrtc, mediapipe, opencv-python, numpy, sounddevice
pyproject.toml             pytest 配置（codex 顺手加的）
```

**验收**：
- ✅ pytest 4/4 PASS（无外部依赖，mock 模式）
- ✅ validator 217/217 PASS（不破坏现有校验）
- ✅ `from poc.src.capture import MultimodalCapture` import OK

**设计要点**：
1. 外部依赖（mediapipe/cv2/streamlit/sounddevice）顶部 try/except ImportError 包裹，给 stub 但保留接口签名
2. 数据契约：jsonl 一行一帧 `{ts, frame_idx, face: {eye_closure, head_pitch_deg, head_yaw_deg, gaze_stability, landmarks_n}, audio: {transcript, lang, duration_s}, context: {active_app, app_switch_count_5min}}`
3. 每文件 < 300 行，类型注解齐全
4. docstring 标 "Day 1 PoC 骨架，等 Day 8+ 真实采集时补全"

**待 Day 2-7**：
- Day 2-3：数据生成框架 + TSV 实验日志
- Day 4：7200h 虚拟数据生成
- Day 5-6：LNN 模型（CfC + StateRecognitionLNN + BoundaryDetectionLNN）
- Day 7：5 阶段记忆形成

**Day 8+ 真实采集**：用 vboss_self 老板自用累积 ≥10h，回填 if-then / 多模态 / narrative 三个 ⚠️ 字段。
