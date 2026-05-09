# Day 7-8 范式跃迁：真实数据持续学习闭环

**更新时间**：2026-05-09（Day 9 Session 补充）  
**分支**：`poc/lnn-validation`

---

## 一、范式跃迁决策（Day 7 拍板）

老板确认抛弃"AI 学 AI"循环，切换至真实数据驱动路线：

| 决策项 | 选择 |
|--------|------|
| 数据来源 | Studio + Air 双摄像头常开采集真实表情 |
| 标注方式 | VLM + py-feat + LLM AI 自动标注，拒绝人工标注 |
| 反馈机制 | 不定期自然语言询问核实（每小时 1-3 次），通过反馈在线 fine-tune |
| 训练位置 | 全部本地（Air 开发 / Studio M2 Max 训练） |
| 隐私边界 | 原始帧 H.264 压缩 24h 滚动删除；衍生信号（AU + embedding + 描述）长期保留 |
| 通知方式 | 飞书 bot |

---

## 二、Day 7 已完成的工作

### 2.1 brow_furrow = 0 根因诊断 + 修复

**根因（4 层）**：
1. `live_verify.py` 默认 `use_pyfeat=False`，走 mediapipe baseline 路径
2. mediapipe 公式 `_BROW_DIST_BASELINE=0.17` 但实际中性值 ~0.22，计算结果恒为 0
3. `face_signals.py` 走静态 `calibrate_to_lnn_expression()`，非 EMA 动态版
4. 训练数据 focused brow attractor = 0.45 与 negative 0.60 重叠，33:1 类别失衡

**修复**：
- `live_verify.py`：默认开启 py-feat（`--no-pyfeat` 可关闭），新增 `--debug-au` 逐帧打印
- 新增 `poc/scripts/diagnose_au.py`：独立 AU 诊断工具（摄像头 → AUCaliper → 统计）
- 新增 `poc/scripts/realign_profiles.py`：批量重对齐 30 个 persona ExpressionProfile
  - focused brow attractor: 0.45 → 0.10-0.20
  - negative brow attractor: 0.60 → 0.70-0.95
  - gap 从 0.15 扩大到 ≥0.61

### 2.2 V2 信号重生成

- 新增 `poc/scripts/regenerate_signals_v2.py`
- 输出目录：`poc/data/synthetic/llm_v2_ltc_v2/`（900 个文件）
- V2 数据特征：focused brow mean=0.11/max=0.35（旧 mean=0.60/max=0.95）

### 2.3 Day 7 V2 训练脚本

- 新增 `poc/scripts/train_day7_v2.py`
- 5 epochs，checkpoint 到 `poc/data/checkpoints/day7_v2/`
- 🟡 **当前状态**：训练中（已运行约 30 分钟，epoch 1 / 5）
- LTC ODE step-by-step 调度是瓶颈，预估 2-3 小时完成

### 2.4 Realtime Capture Daemon V0.1

新建 `poc/src/realtime/` 目录，6 个文件：

| 文件 | 功能 | 行数 |
|------|------|------|
| `daemon_config.py` | TOML 配置 + FeedbackConfig（查询频次 1-3/小时）+ preflight 自检 | 185 |
| `capture_worker.py` | cv2 摄像头采集线程（指数退避重连）| 124 |
| `signal_pipeline.py` | **py-feat 唯一信号源（LNN-FIRST）**：AU + 情绪 + img2pose 头姿 | 122 |
| `storage_manager.py` | H.264 分时段录制 + JSONL append + 24h/365d 滚动清理 | 190 |
| `daemon_main.py` | 编排 + HTTP API(9876) + TCC 摄像头授权预检 | 195 |
| `__init__.py` | 包入口 | 5 |

**HTTP API**：
- `GET /health` → `{status, uptime_s, frames_processed, last_frame_ts}`
- `GET /signals/recent?seconds=N` → 最近 N 秒衍生信号列表
- `POST /control/stop` → 优雅停止

**FeedbackConfig（Day 9+ 接入点）**：
```toml
[feedback]
query_min_per_hour = 1
query_max_per_hour = 3
query_confidence_threshold = 0.65
dnd_apps = ["zoom", "loom", "facetime", "obs"]
```

### 2.5 双摄像头交叉验证上线

| 设备 | 状态 | frames_processed | py-feat |
|------|------|-----------------|---------|
| **Air** | ✅ 运行中 | ~4900+/uptime≈100min | ✅ AU + expression（20 AUs 全出值）|
| **Studio** | ✅ 运行中，AU 全激活 | ~250+/uptime≈5min | ✅ resmasknet + XGB AU 20 channels |

**Day 9 Session 补充说明**：Studio py-feat 踩坑排查记录：
- 根因 1：`emotion_model=None` / `facepose_model=None` → py-feat 0.6.2 `get_pretrained_models()` 直接 `raise ValueError`
- 根因 2：`emotion_model="svm"` 需要 `July4_emo_SVM.pkl`，该文件未随包安装，Studio 无外网→60s TCP timeout 阻塞
- 修复：改用 `emotion_model="resmasknet"`（本地 .pth），rsync Air→Studio 三个大文件（img2pose 162MB + facenet 107MB + ResMaskNet 528MB）
- 当前配置：`Detector(au_model="xgb", emotion_model="resmasknet")`，其他参数取 py-feat 默认值

**启动命令（两台都需要在 Terminal.app 里运行，非 SSH）**：
```bash
# Air
cd /tmp/lnmem-fix/LnMem
PYTHONPATH=. .venv-poc/bin/python -m poc.src.realtime.daemon_main --device air --camera 0

# Studio  
cd /Users/fu-121/lnmem-poc/LnMem
PYTHONPATH=. .venv-poc/bin/python -m poc.src.realtime.daemon_main --device studio --camera 0
```

**信号格式**（Day 9 更新：head_pose 改为 img2pose 3DoF）：
```json
{
  "ts": "2026-05-09T02:00:51Z",
  "device": "air",
  "frame_id": 493,
  "face_detected": true,
  "raw_au": {"AU04": 0.18, "AU12": 0.05, ...},
  "calibrated_au": {"AU04": 0.205, "AU12": 0.052, ...},
  "expression": {"smile_intensity": 0.056, "brow_furrow": 0.205, ...},
  "head_pose": {"yaw_deg": -4.7, "pitch_deg": 1.9, "roll_deg": 0.3},
  "session_id": "uuid"
}
```

---

## 三、Gemini 调研报告核心结论

完整报告：`/tmp/gemini_research_lnn_realtime.md`（本地）

### 推荐技术路线（Day 8-13）

| 模块 | 推荐方案 | LNN-FIRST 映射 |
|------|---------|---------------|
| VLM 自标注 | 本地 Qwen2-VL（MLX） + CoT Prompting | AU intensities → I(t) 输入 LTC |
| 主动询问触发 | AU burst + VLM 不一致时触发，每小时 1-3 次 | modAL / Baal 不确定性估计 |
| 反馈训练 | 连续 DPO，回答映射为 Δ(V/A)，奖励作用于 τ 偏移 | TRL / RL4LMs |
| 持续学习 | 冻结视觉主干 + 微调 LTC 映射层 + <100 样本 replay buffer | Avalanche + PEFT/LoRA |
| 个性化基线 | 7 天滚动中性 AU 均值，ΔAU 作为 I(t) | AUCaliper EMA（已实现）|
| 隐私 | RAM 内处理，仅落盘 EventTensor，关键帧高斯模糊 24h 后删 | 已实现（V0.1 daemon）|

**最大风险**：常驻电脑前表情极其细微（Micro-expressions），VLM 大量输出 Neutral → LTC 神经元状态停止转移（"死水"）。需在 τ 偏置层放大微表情敏感度。

---

## 四、后续 TODO（按优先级）

### P0 — 训练完成后立即

- [ ] **训练 Day 7 v2 完成** → 检查 `checkpoint_day5.pt` val_acc（目标 ≥70%）
- [ ] **live_verify v3 重测**：运行 `poc/scripts/live_verify.py --debug-au` 10 分钟，确认 emotional_negative 能触发
- [x] ✅ **Studio py-feat 双摄像头上线** — AU 20 channels 全激活（brow_furrow/smile_intensity 实时输出）

### P1 — Day 9（VLM 自标注 Pipeline）

- [ ] **部署 Qwen2-VL（MLX）到 Studio**
  - Studio M2 Max 128GB，Qwen2-VL-7B-Instruct 完全装得下
  - `mlx_lm.convert` 量化到 Q4/Q8
  - 接入 Studio daemon 的 pipeline，每 10 帧触发一次 VLM 推断
- [ ] **VLM Prompt 设计（CoT 模式）**
  - 参考 Facial-R1 / ExpLLM 论文：先描述 AU 肌肉运动 → 再推导情绪连续体
  - 强制输出 `{"au_description": "...", "valence": 0.3, "arousal": 0.7, "confidence": 0.8}`
- [ ] **自标注结果写回 JSONL**：在 `expression` 字段追加 `vlm_label: {...}`

### P2 — Day 9-10（主动询问 + 反馈）

- [ ] **飞书 bot 询问模块**
  - 触发条件：AU burst（brow_furrow > 0.7 持续 5s）OR VLM confidence < 0.65
  - 上下文感知：检测前台 App，排除 zoom/loom/facetime
  - 询问模板："老板，刚才我注意到你皱眉头了，是有什么让你不爽吗？"
- [ ] **反馈解析 → Δ(AU Attractor) 调整**
  - LLM 解析老板回答 → Valence/Arousal 真实 Δ 值
  - 更新对应 ExpressionProfile 的 state_offset（在线学习）

### P3 — Day 10-13（在线 Fine-tune + 闭环）

- [ ] **LTC Adapter 层**：冻结主干，仅微调 AU→LTC 映射层（LoRA 方案）
- [ ] **Replay Buffer**：维护 <100 个老板极端状态样本，每日混合 fine-tune
- [ ] **交叉验证逻辑**：Air + Studio 按 ±5s 时间窗口对齐，差异 > 阈值触发"遮挡/光照异常"告警
- [ ] **7 天滚动基线更新**：每天凌晨计算过去 7 天中性帧均值，更新 AUCaliper baseline

### P4 — Day 13 Demo 目标

- [ ] 老板眉头紧锁 5 秒 → 飞书询问 → 收到回复 → LTC 状态负向偏移可视化
- [ ] 双摄像头同帧交叉验证通过率 > 80%（同一表情，两台摄像头结果一致）
- [ ] 主动询问准确率 > 80%（对触发的样本，老板确认系统判断正确）

---

## 五、关键文件路径

```
poc/
├── src/realtime/           # Day 8 新增：Daemon V0.1（Day 9 LNN-FIRST 更新）
│   ├── daemon_config.py    # 配置（enable_mediapipe=False 已弃用）
│   ├── capture_worker.py   # 摄像头采集
│   ├── signal_pipeline.py  # LNN-FIRST：py-feat 唯一信号源（MediaPipe 已删除）
│   ├── storage_manager.py  # H.264 + JSONL 存储
│   └── daemon_main.py      # 主入口（含 TCC 摄像头授权预检）
├── src/vision/
│   ├── au_extractor.py     # AUExtractor.extract() → 3-tuple (au, emo, pose)
│   └── feature_calibrator.py  # AUCaliper（EMA + gamma + 相对归一化）
├── scripts/
│   ├── diagnose_au.py      # AU 诊断工具
│   ├── realign_profiles.py # 30 persona 重对齐
│   ├── regenerate_signals_v2.py  # V2 信号重生成
│   └── train_day7_v2.py    # Day 7 V2 训练（5 epoch）
├── data/personas/expression_profiles/  # 30 个 vboss 已重对齐
└── data/checkpoints/day7_v2/  # 🟡 训练中
```

**Studio 路径**：`/Users/fu-121/lnmem-poc/LnMem/`（daemon 已同步）

---

## 六、踩坑总结（Day 8-9，6 条铁律）

### 坑 1：py-feat 0.6.2 `__init__` 同步下载全部模型（非懒加载）

**现象**：Studio daemon 启动 `frames_processed=0` 卡死数分钟，无 Python 报错，只有 `/health` 请求流正常。  
**根因**：`Detector.__init__()` → `get_pretrained_models()` → 对每个模型文件调 `download_url()` → `_get_redirect_url()` → TCP connect → **60 秒超时**。Studio 出口网络受限，每个文件都超时才继续。  
**铁律**：`Detector()` 调用前必须确保所有模型文件已在磁盘。诊断命令：`ls feat/resources/` 对比 `model_list.json`。

### 坑 2：`emotion_model=None` / `facepose_model=None` = 立即 ValueError，不是跳过

**现象**：传 `Detector(facepose_model=None, identity_model=None)` 报 `ValueError: facepose_model must be one of [...]`。  
**根因**：py-feat 0.6.2 对 `None` 参数立即 `raise ValueError`，没有"跳过此模型"语义。  
**铁律**：必须传合法字符串（`"img2pose"` / `"resmasknet"` 等）或完全省略让 py-feat 取默认值。

### 坑 3：`emotion_model="svm"` 依赖 `July4_emo_SVM.pkl`，pip 不发布这个文件

**现象**：有网机器正常，无网 Studio 卡死 60 秒。  
**根因**：该 pkl 文件在 GitHub release URL，pip install 不下载。首次使用触发下载。  
**铁律**：用 `emotion_model="resmasknet"`（本地 528MB .pth，rsync 一次永不再下载，性能也更好）。

### 坑 4：LNN-FIRST 架构违反：py-feat + MediaPipe 混用破坏信号一致性

**现象**：`signal_pipeline.py` 同时跑 MediaPipe FaceMesh 算头姿（yaw/pitch）+ py-feat 算 AU/情绪。  
**根因**：两套面部分析管道的 face alignment 算法不同，坐标系不一致。同一帧可能一个检测到人脸另一个没有，导致字段不一致。额外消耗 RAM/CPU。  
**铁律**：选择 py-feat 后，**所有面部信号必须从同一个 `detect_image()` 的 fex 里提取**。`fex["Pitch"/"Roll"/"Yaw"]` = img2pose 3DoF 头姿，比手写的 MediaPipe 两点公式更准（3DoF vs 2DoF）。

### 坑 5：macOS `kill -3` = SIGQUIT → 进程立即终止

**现象**：想 dump Python 线程堆栈调试 `frames_processed=0`，`kill -3 <PID>` 后进程消失。  
**根因**：Linux `kill -3` 会让 Python 打印 traceback；macOS `kill -3` 默认行为是终止进程，无 core dump。  
**铁律**：macOS Python 调试用 `kill -SIGUSR1`（如果已注册）或加 `-u` 参数防缓冲；不要用 `kill -3`。

### 坑 6：摄像头 TCC 授权必须在 Terminal.app 主线程，不能在 SSH/Background 线程

**现象**：`daemon_main.py` 用 `run_in_background` 启动 → OpenCV `can not spin main run loop from other thread`。  
**根因**：macOS TCC 摄像头权限弹窗只能在主线程 run loop 里触发，SSH session 无 GUI 上下文。  
**铁律**：daemon 启动命令必须用 `osascript -e 'tell application "Terminal" to do script "..."'` 触发 Terminal.app 新窗口，不能用 SSH 后台。

---

## 七、下个 Session 接手指引

1. **检查训练是否完成**：
   ```bash
   ls /tmp/lnmem-fix/LnMem/poc/data/checkpoints/day7_v2/
   ```
2. **如未完成，等待或重启训练**：
   ```bash
   cd /tmp/lnmem-fix/LnMem && PYTHONPATH=. .venv-poc/bin/python -u poc/scripts/train_day7_v2.py
   ```
3. **live_verify v3 重测**（需 val_acc 出炉后）：
   ```bash
   cd /tmp/lnmem-fix/LnMem && PYTHONPATH=. .venv-poc/bin/python poc/scripts/live_verify.py --debug-au
   ```
4. **检查双 daemon 状态**：
   ```bash
   curl -s http://127.0.0.1:9876/health        # Air
   ssh studio "curl -s http://127.0.0.1:9876/health"  # Studio
   ```
   ✅ **Studio py-feat 已完成**：emotion_model=resmasknet，AU 20 channels 全激活。无需再执行任何安装步骤。

5. **确认 LNN-FIRST head_pose 3DoF**：日志 `head_pose` 字段必须有 `roll_deg`（旧 MediaPipe 路径只有 2 字段）：
   ```bash
   curl -s "http://127.0.0.1:9876/signals/recent?seconds=10" | python3 -c "import sys,json; s=json.load(sys.stdin); print(s[0]['head_pose'])"
   ```

6. **下一步主要任务**：部署 Qwen2-VL（MLX）到 Studio，构建 VLM 自标注 pipeline（P1）
