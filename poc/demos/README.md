# PoC Demos — 真实采集 smoke 测试与录屏

> 用途：在老板/团队真实环境（macOS + 摄像头 + 麦克风）跑 smoke 验证 capture 链路硬门槛。
> Day 14 演示录屏也归这里。

## 文件清单

| 文件 | 用途 |
|------|------|
| `smoke_realtime.py` | 4 阶段 smoke：MediaPipe静态图 / osascript上下文 / sounddevice 16kHz / cv2+MediaPipe 摄像头 30fps |
| `results/2026-05-05-day2-smoke.jsonl` | Day 2 收官实测原始数据（jsonl 一行一项，永久 tracked） |

## 怎么跑

**前置**（一次性）：
1. `brew install portaudio`（sounddevice 系统依赖）
2. `.venv-poc/bin/pip install -r poc/requirements-day1.txt`
3. **首次必须在 macOS 自带 Terminal.app 里跑**（不是 iTerm/Codex），让 macOS TCC 弹出
   摄像头/麦克风授权框，老板点"允许"。授权后 TCC 记录持久化，之后任意终端都可跑。

**全套 smoke**（首次会触发 2 次系统授权框：麦克风 + 摄像头）：

```bash
.venv-poc/bin/python poc/demos/smoke_realtime.py all
```

**只跑某一项**（已授权过，定向验证）：

```bash
.venv-poc/bin/python poc/demos/smoke_realtime.py mediapipe_static    # 不需权限
.venv-poc/bin/python poc/demos/smoke_realtime.py active_app          # 不需权限
.venv-poc/bin/python poc/demos/smoke_realtime.py audio               # 麦克风
.venv-poc/bin/python poc/demos/smoke_realtime.py camera              # 摄像头
```

结果落盘：`/tmp/lnmem_smoke_results.jsonl`（脚本默认）。归档时复制到
`results/YYYY-MM-DD-<note>.jsonl`，git tracked。

## Day 2 收官硬证（2026-05-05）

老板真实环境（MacBook Air M-series / Apple M5 GPU / Tahoe）跑通：

| README §时间盒硬门槛 | 实测 | 状态 |
|--------------------|------|------|
| 30fps 摄像头 | 25.5fps / 60 帧 / 2.36s（95% 人脸识别 57/60） | ✅ |
| MediaPipe FaceMesh | 478 landmarks / refine_landmarks / Apple M5 Metal GPU + XNNPACK delegate | ✅ |
| 16kHz 麦克风 | 16000 samples @ 16kHz / rms=0.0129（真采环境音） | ✅ |
| macOS 上下文 | osascript active_app='Codex' | ✅ |

原始 jsonl：[`results/2026-05-05-day2-smoke.jsonl`](results/2026-05-05-day2-smoke.jsonl)

## 重要陷阱

**mediapipe 0.10.20+ 砍掉了 `mp.solutions` namespace** 改成 Tasks API。骨架代码用的是
`mp.solutions.face_mesh.FaceMesh`，所以 [requirements-day1.txt](../requirements-day1.txt)
钉了 `mediapipe<0.10.20`。如果未来要升级到 Tasks API，需要重写 `face_signals.py` 用
`mp.tasks.vision.FaceLandmarker` + 下载 `face_landmarker.task` 模型文件。

**macOS TCC 摄像头权限按 app 授权** —— OpenCV 在没授权的进程里打开摄像头会立即失败
（`OpenCV: not authorized to capture video (status 0)` + `camera failed to properly
initialize`），不像 GUI app 会等待授权框。**首次授权必须在常用终端启动一次**。
之后 cv2 + MediaPipe 直接 `cv2.VideoCapture(0)` 即可。
