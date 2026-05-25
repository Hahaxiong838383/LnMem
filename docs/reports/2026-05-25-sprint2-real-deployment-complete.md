# Sprint 2 实战部署完整复盘（5/25 02:50-09:30，~7h 团队模式 + 川哥并肩实测）

> **意义**：ADR-028 v0.1 Sprint 2 从 spec 文档 → 真实跑通的整夜实战。
> 川哥三选 A 钦定方案 100% 落地 + Studio 公网部署 + 第一条真反馈 jsonl + 旧 engine 静默化。
> 详细 spec：[`docs/decisions/029-sprint2-intent-classifier-daf-studio-pipeline.md`](../decisions/029-sprint2-intent-classifier-daf-studio-pipeline.md)

---

## 1. 终态系统全景

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Air M5 (川哥日常工作 + daemon)                                          │
│                                                                         │
│  📷 camera 30Hz → mp 2-worker py-feat (1.07 fps)                        │
│     ↓                                                                   │
│  ✅ LNN backbone (phase2 ckpt sha8=59cd1d9c, 5/22 11 轮训练)             │
│     window=8 帧 → 8s 暖机（vs 修复前 60帧 = 56s）                       │
│     ↓                                                                   │
│  HTTP :9876 /health /signals/recent /feedback/stats                     │
│  HTTP :9877 /intent_feedback (POST)                                     │
│     ↓                                                                   │
│  📊 Streamlit UI http://localhost:8520 (PID 9512)                       │
│     实时显示 8 类状态 + 4 按钮反馈                                       │
│     ↓                                                                   │
│  🔔 notify_watcher.sh 后台守护 (PID 9484)                                │
│     osascript 3 按钮系统弹窗 + 反馈落盘                                  │
│     ↓                                                                   │
│  ✅ ~/.lnmem/data/intent_feedback/YYYY-MM-DD.jsonl                       │
│     含 hidden_state_window 10s/300 帧（gemini H1 钦定）                  │
└─────────────────────────────────────────────────────────────────────────┘
                        ↑ ckpt pull (10min cron)            ↓ rsync data (hourly cron)
┌─────────────────────────────────────────────────────────────────────────┐
│ Studio M2 Max (公网 ssh studio-pub / iBRO ssh studio-ibro)              │
│                                                                         │
│  ✅ uv 0.11.16 + python 3.11.15 + venv (torch 2.12 / numpy 2.4 / MPS)   │
│  ✅ ~/lnmem-poc 仓 558MB（Air rsync）                                    │
│  ✅ ~/lnmem-daemon-sync 14 天 daemon jsonl 659MB                         │
│  ✅ ~/lnmem-train/daemon_au_cache.npz 19MB (daemon_dataset_cli 构)       │
│  ✅ ~/lnmem-train/ckpt/ 输出位                                            │
│                                                                         │
│  03:00 cron → studio_nightly_train.sh                                   │
│    检查 intent_feedback ≥ 50 条 → train_intent_classifier.py            │
│    → ckpt 落 ~/lnmem-train/ckpt/intent_head_latest.pt                   │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. GitHub commit 时间线（凌晨到早晨）

| commit | 时间 | 内容 |
|--------|------|------|
| `578baf3` | 03:30 | Sprint 2 一轮（codex + cc 主线，19 测试过） |
| `2e260c6` | 04:00 | Sprint 2 retrofit（gemini 8 finding 整改，24 测试过） |
| `64c66e8` | 08:00 | 自动加载 phase2 ckpt + LNMEM_INTENT_HEAD_CKPT 支持 |
| `0cc73a5` | 08:10 | UI: ws → HTTP polling 修复 |
| `0f7a3f3` | 08:20 | window 60→8 + 8 类标签 + notify_watcher.sh |
| `d900e66` | 08:42 | osascript 4 按钮 → 3 按钮（系统限制） |
| `d3fb96e` | 08:45 | predicted 字段 schema 兼容（强制 3 类） |
| `ffc055b` | 09:20 | 修 RateLimiter max=0 死循环 + enable_prompt=False 旁路 disable |

## 3. 实证里程碑

| 维度 | 实证 |
|------|------|
| ADR-028 异步监督学习闭环 | ✅ **首次真数据流通**（川哥点 [→正] → 1s 内落 jsonl）|
| gemini H1 钦定 hidden_state_window | ✅ **真落盘 64-dim phase2 hidden 向量**（norm ~0.97）|
| phase2 backbone 复用 5/22 11 轮成果 | ✅ daemon /health `ckpt_sha8=59cd1d9c` 持续输出 |
| Studio 公网部署 | ✅ ssh studio-pub (cli4.hahaxiong.cc:22022 跳板)|
| Studio 14 天数据 | ✅ 659MB jsonl rsync 完整 |
| Studio dtaware 端到端 | ✅ 10 epoch / loss 0.20→0.04 / dt-aware 数值正确（actual_dt=0.994）|
| 暖机时间 | 56s → **8s**（_DEFAULT_WINDOW 60→8）|
| 弹窗反馈 | ✅ osascript 3 按钮 + 60s gave up + 180s cooldown |
| 旧 ProactiveQueryEngine 静默 | ✅ enable_prompt=False 旁路 |

## 4. 五大踩坑教训（按事故等级排序）

### 4.1 配置值合法区间陷阱（事故等级 P0 - daemon 反复重启 30 分钟）

**根因**：commit `d3fb96e` 把 daemon.toml `query_max_per_hour=0` 想关旧弹窗，但 `RateLimiterConfig` 校验 `must be > 0`，daemon 启动炸 → supervisor 检测 health fail → osascript 启 Terminal 重启 → 再炸 → 死循环。

**正确套路**：
- 配置「禁用」语义不要靠 `=0`
- 用合法占位值（如 `=1`）+ 代码层 flag 旁路（如 `enable_prompt=False`）
- 改任何 toml/json 配置先查代码中对应字段的校验逻辑

**修复**：commit `ffc055b`

### 4.2 osascript display dialog 3 按钮硬限制（事故等级 P1 - 川哥看不到弹窗）

**根因**：notify_watcher 用 `buttons {跳过, →负, →无, →正}` 4 按钮，osascript 报 `-50 最多允许使用三个按钮`，dialog 完全弹不出来。我看到 log 里 `user choice:` 空，误以为川哥没点。

**正确套路**：
- macOS display dialog 限 1-3 按钮
- 「跳过」靠 ESC + `giving up after N` 内置超时
- 超过 3 选项要用 `choose from list` 或 SwiftUI

**修复**：commit `d900e66`

### 4.3 滑窗大小没匹配 daemon 实际 fps（事故等级 P1 - 56s 暖机）

**根因**：`_DEFAULT_WINDOW = 60` 是给 30Hz 设计的（2 秒窗口），daemon 实际 1.07fps → 60 帧 = **56 秒暖机**！川哥实测「做笑脸几十秒没反馈」根因。

**正确套路**：
- 滑窗大小必须 = `期望窗口秒数 × 实际 fps`
- 写 inferencer 时考虑实际部署 fps，不能 hardcode
- 用 env `LNMEM_INFER_WINDOW` 允许运行时覆盖

**修复**：commit `0f7a3f3`，60 → 8

### 4.4 cc 假设接口形态错（事故等级 P1 - UI 永远 "等待 ws 数据"）

**根因**：写 streamlit UI 时假设 daemon 9876 是 WebSocket，实际是 HTTP（BaseHTTP），连接 ws 直接 404。

**正确套路**：
- 写 client 代码前必须 `curl http://X:port/path` 实测一下 endpoint 类型
- 别从「daemon 名字像 inference server」推断「应该是 ws」

**修复**：commit `0cc73a5`

### 4.5 schema 锁死反馈被拒（事故等级 P0 - 反馈全部 invalid_payload）

**根因**：daemon `/intent_feedback` 校验 `predicted ∈ {positive, negative, none}`（ADR-028 3 类），但 phase2 backbone 输出的是 8 类（transitioning/focused/...）。UI/watcher 直接传 8 类 → 全部 invalid_payload 落不进 jsonl。

**正确套路**：
- 不同层级的 schema 命名要明确（`predicted_state` 8 类 vs `predicted` 3 类）
- 跨 schema 转换时强制兜底（fallback `"none"` + 真实 state 进 `note` 字段保留）

**修复**：commit `d3fb96e`

## 5. macOS 环境工具链坑（次要但反复踩）

| 坑 | 修复 | mycc memory |
|----|------|-------------|
| macOS 无 `timeout` 命令 | 用 `osascript ... giving up after N` / `nohup` / Monitor 工具替代 | 新增 `feedback_macos_no_timeout.md` |
| `uv venv` 默认不带 `pip` | 用 `uv pip install --python .venv/bin/python` | 新增 `feedback_uv_venv_no_pip.md` |
| `uv pip install` 跨海 PyPI torch 卡 20min | `--index-url https://pypi.tuna.tsinghua.edu.cn/simple` 秒装 | 新增 `feedback_uv_tsinghua_mirror.md` |
| Streamlit welcome email 卡住 | 写 `~/.streamlit/credentials.toml` 永久禁用 | 新增 `feedback_streamlit_welcome_skip.md` |
| GitHub HTTP/2 framing layer error | Fallback rsync 推代码（绕 GitHub）| 已记 `feedback_macos_openrsync_limits.md` 同类 |
| macOS openrsync 不支持 `--info=stats2` | 已记 mycc memory | 已存在 |
| osascript dialog 4+ 按钮限制 | 改 3 按钮 + ESC 跳过 | 新增 `feedback_osascript_dialog_button_limit.md` |
| Air SSH 别名不存在（DHCP 动态 IP） | Air 主动 pull 模式 + studio 不需要知道 Air | 新增 `feedback_air_dhcp_no_ssh_alias.md` |
| Studio lnmem-poc 不是 git 仓 | rsync 推代码（绕 GitHub HTTP/2）| 已沉淀 |

## 6. 后续 TODO（按优先级）

### P0（5/25 当天）
- [ ] 川哥用 UI 给反馈，目标 50 条
- [ ] 装 LaunchAgent: `./scripts/install_launchd_air.sh` + `ssh studio-ibro "cd ~/lnmem-poc && ./scripts/install_launchd_studio.sh"`
- [ ] 写 mycc memory 9 篇教训文件

### P1（5/26-5/27）
- [ ] 攒够 50 条 → Studio 03:00 cron 自动训第一个 intent_head ckpt
- [ ] 或川哥手动 `ssh studio-ibro "~/lnmem-poc/scripts/studio_nightly_train.sh --force"`
- [ ] daemon 软重启加载新 head ckpt
- [ ] 验证 `intent_head_enabled=true` + `intent_pred` 不再是 None

### P2（5/28-6/3 M0 demo 收尾）
- [ ] 24h 学习曲线（每天准确率 +X pp）
- [ ] Studio USB 摄像头当 hold-out 验证节点（gemini D1 防 train/test 同摄像头 overfit）
- [ ] 录屏 + 川哥外部分享物料

## 7. 给 AI 同事的接手指引（30 秒）

进入项目第一步：
1. 读本文件 §1 系统全景
2. 看 `ROADMAP.md` 顶部当前阶段
3. `curl http://127.0.0.1:9876/health` 看 daemon 状态
4. `wc -l ~/.lnmem/data/intent_feedback/$(date +%Y-%m-%d).jsonl` 看当天反馈量
5. `tail -20 /tmp/notify_watcher.log` 看弹窗事件

## 8. 川哥钦定的「最小可用单位」实证

> **「我点反馈 1 秒内有 hidden_state_window 落 jsonl」= Sprint 2 最小可用单位**

5/25 08:44:03 实证：
```json
{
  "ts_ns": 1779669540245711104,
  "corrected": "positive",                              ← 川哥真意图
  "hidden_state_window": [{"h": [-0.97, -0.97, ...]}]   ← phase2 backbone 真特征
}
```

**这一条 jsonl = Sprint 2 团队模式 7h 实战的全部价值集中物**。

---

**署名**：cc（Claude）+ codex（gpt-5.3-codex high · 2 轮）+ gemini（3-flash-preview · 8 finding）
**川哥实测同行 + 4 次 UX bug 反馈纠偏（每次 < 10 分钟修复）**
**落盘时间**：2026-05-25 09:30 凌晨 + 早晨连续 7h 团队模式收工
