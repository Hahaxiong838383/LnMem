# ADR-029: Sprint 2 — Intent Classifier + DAF 反馈闭环 + Studio 训练管线

> **决策者**：川哥 2026-05-24 凌晨钦定 Q1=A / Q2=A / Q3=A
> **状态**：accepted（团队模式开干）
> **关联**：ADR-028 LNN 角色重定义（直接落地） · 整合 gemini 8 finding 整改
> **生效**：5/24 - 6/3（10 天到 M0 demo）

---

## 1. 川哥钦定三参数

| # | 维度 | 钦定值 | 落地脚本 |
|---|------|--------|---------|
| Q1 | 数据流（Air→Studio） | **每小时 rsync 增量，排除 raw_video，no --delete** | `scripts/sync_daemon_to_studio.sh` |
| Q2 | Studio 角色 | **纯训练**：夜间 03:00 训练 intent_head，Air 主动 pull ckpt | `scripts/studio_nightly_train.sh` + `scripts/pull_ckpt_from_studio.sh` |
| Q3 | Sprint 2 节奏 | **立即启动（5/24）**：UI 和 classifier 同步上，不等数据 | 见下方 timeline |

## 2. 系统架构（v0.1 Sprint 2）

```
  ┌──────────────────────────────────────────────────────────────┐
  │ Air M5 (daemon 主战场)                                       │
  │                                                              │
  │  📷 camera 30Hz                                              │
  │     ↓                                                        │
  │  capture_worker → mp 2-worker py-feat pool (1.07fps)         │
  │     ↓                                                        │
  │  signal_pipeline → LNN inferencer (60 帧滑窗)                │
  │     ├→ 推断结果 + intent_probs (3-class)                     │
  │     │     ↓                                                  │
  │     │   ws://9876 → Streamlit UI 弹卡                        │
  │     │              ↑                                         │
  │     │            POST /intent_feedback :9877                 │
  │     │                                                        │
  │     └→ jsonl 落盘 ~/.lnmem/data/signals_YYYY-MM-DD.jsonl     │
  │           + intent_feedback/YYYY-MM-DD.jsonl                 │
  │                                                              │
  │  📥 ckpt pull cron (每 10min)                                │
  │     pull_ckpt_from_studio.sh → ~/.lnmem/ckpt/                │
  │     daemon 监 mtime → 软重启 inferencer (新旧并行 1s 过渡)    │
  └──────────────────────────────────────────────────────────────┘
                   ↑ ckpt          ↓ rsync data (hourly)
                                   
  ┌──────────────────────────────────────────────────────────────┐
  │ Studio M2 Max (纯训练算力)                                   │
  │                                                              │
  │  📂 ~/lnmem_data/  ← rsync from Air (hourly)                 │
  │                                                              │
  │  ⏰ cron 03:00 → studio_nightly_train.sh                     │
  │     1. git pull --rebase                                     │
  │     2. 检查反馈条数 >= 50（否则跳过）                          │
  │     3. .venv/bin/python train_intent_classifier.py           │
  │        --data-dir ~/lnmem_data                               │
  │        --out-ckpt ~/lnmem_ckpt/intent_head_YYYYMMDD.pt       │
  │     4. ln -sf 维护 intent_head_latest.pt                     │
  │                                                              │
  │  📂 ~/lnmem_ckpt/  ← Air 主动 pull                            │
  └──────────────────────────────────────────────────────────────┘
```

## 3. Gemini 8 Finding 整改（必做）

### 3.1 高风险整改

| # | gemini 风险 | 整改动作 | 落地位置 |
|---|------------|---------|---------|
| H1 | LNN 序列学习与异步反馈断裂 | **intent_feedback 落盘必含 10s 前序 hidden state 窗口**（Studio BPTT 训序列动力学不是瞬时统计） | codex Sprint 2-B retrofit |
| H2 | None 类塌陷 | **Dynamic Focal Loss** + Positive/Negative 10x 过采样（cost-sensitive 0.5/1.0 不够） | train_intent_classifier.py (cc 后续) |
| H3 | 软重启状态跳变 | **平滑切换**：新旧 inferencer 并行 1s，hidden state 线性加权过渡 | daemon ckpt reload 逻辑 |

### 3.2 设计漏洞整改

| # | 漏洞 | 整改 |
|---|------|------|
| D1 | py-feat 版本两端不一致 | requirements.txt 锁版本 + Studio CI 校验 `py-feat==0.6.2` |
| D2 | 反馈记忆污染 | UI 反馈窗口 ≤ 120s（streamlit_ui FEEDBACK_WINDOW_SEC 已实现） |
| D3 | 缺硬负反馈 | UI 加一个「🚨 烦躁」按钮，写入 corrected="negative_high_priority" |
| D4 | rsync 竞态 | **先 mv 归档再 sync**：daemon 每小时 mv 当前 jsonl 到 `.archive`，rsync 只同步 .archive |
| D5 | 64-dim 表达浪费 | intent_head 中间加 L2 正则 + Dropout 0.1（已在 spec） |

### 3.3 川哥可用性整改

| # | 盲区 | 整改 |
|---|------|------|
| U1 | 反馈疲劳 | UI 默认静默模式（置信度 >= 0.8 才弹卡，已实现 SILENT_CONFIDENCE_THRESHOLD） |
| U2 | 隔夜生效迟钝感 | demo 期增加 **手动「立即重训」按钮**，UI 触发 Studio 训练（绕 03:00 cron） |
| U3 | 隐私焦虑 | UI 只显示推断标签，不显示原始 AU 数值；spec 写明 raw_video 永不出 Air |

## 4. 接口契约

### 4.1 WebSocket 输出（inference server → UI）

```json
{
  "ts_ns": 1717000000000000000,
  "intent_pred": "positive | negative | none | warmup | error",
  "intent_confidence": 0.87,
  "intent_probs": {"positive": 0.87, "negative": 0.08, "none": 0.05},
  "margin": 0.79,
  "entropy": 0.31,
  "ckpt_sha8": "a1b2c3d4",
  "signal_path": "primary"
}
```

### 4.2 HTTP /intent_feedback（UI → daemon → jsonl）

```json
POST http://127.0.0.1:9877/intent_feedback
{
  "ts_ns": 1717000000000000000,
  "predicted": "positive",
  "corrected": "negative",
  "ui_session": "uuid",
  "note": "optional 文字补充"
}
Response: { "accepted": true, "stored_at": "2026-05-24T03:00:00Z" }
```

落盘 schema（`~/.lnmem/data/intent_feedback/YYYY-MM-DD.jsonl`，**包含 H1 整改**）：

```json
{
  "ts_ns": ...,
  "predicted": "...", "corrected": "...",
  "ui_session": "...",
  "note": "...",
  "hidden_state_window": [   // ← gemini H1 整改：10s 前序窗口
    {"ts_ns": ..., "h": [64-dim float array]},
    ... (按 daemon fps 采样，~10-30 个点)
  ],
  "au_window": [             // 同窗口的原始 AU 特征（训练用）
    {"ts_ns": ..., "au": [18-dim]},
    ...
  ],
  "recorded_at": "ISO-8601"
}
```

### 4.3 ckpt 同步约定

- Studio 训完落 `~/lnmem_ckpt/intent_head_YYYYMMDD-HHMMSS.pt` + `intent_head_latest.pt` symlink
- Air `pull_ckpt_from_studio.sh` 每 10min cron 比 readlink，新则 rsync + 原子换 symlink
- daemon 监 `~/.lnmem/ckpt/intent_head_latest.pt` mtime，变化触发软重启

## 5. Sprint 2 Timeline（5/24-6/3）

| 阶段 | 日期 | 任务 | 负责 |
|------|------|------|------|
| **Day 1 凌晨** | 5/24 02:30-04:00 | spec + codex 一轮 + UI v0 + gemini 评审 | cc + codex + gemini |
| **Day 1 早** | 5/24 起床后 | 川哥验收，cc 派 codex 二轮 retrofit（gemini H1/H2/H3） | 川哥拍 |
| **Day 2-3** | 5/25-5/26 | **冷启动期**：daemon + UI 跑通，川哥用，攒反馈数据 | 川哥 + cc 联调 |
| **Day 4** | 5/27 | 第一次 Studio 训练（手动 `--force` 触发，看 baseline） | cc Studio 执行 |
| **Day 5-7** | 5/28-5/30 | 反馈数据积累 + 模型迭代 + UI 改进 | 全员 |
| **Day 8-9** | 5/31-6/1 | Studio cron 接管自动训练 | cc 部署 |
| **Day 10** | 6/2-6/3 | M0 demo 收尾：录屏 + 24h 学习曲线 + 川哥外部分享物料 | cc + 川哥 |

## 6. 验收门（M0 demo）

| 门 | 指标 | 通过线 |
|----|------|--------|
| 实时性 | ws 推断到 UI 显示延迟 | < 200ms |
| 反馈延迟 | UI 点击到 jsonl 落盘 | < 500ms |
| 训练有效性 | 第 7 天 intent_pred 准确率 vs 第 1 天 | +20pp 以上（在 100+ 反馈样本上） |
| 川哥体验 | 1 周用下来不嫌烦 | 自主评分 ≥ 4/5 |
| 系统稳定 | daemon 7×24 不崩 + ckpt 热重载零跳变 | 0 重启 / 0 闪烁 |

## 7. 不在 Sprint 2 范围

- ❌ Tauri menu bar widget（留 M1）
- ❌ Studio 接 USB webcam 当二号节点（留 v0.2）
- ❌ shared_memory 多 worker mp pickle 优化（留 v0.2）
- ❌ OpenFace 替代 py-feat 评估（留 v0.2）
- ❌ 14 天回忆界面（川哥钦定不做，传统记忆系统能做）

## 8. 关联

- **依赖 ADR**：ADR-028（LNN 角色重定义）
- **取代**：旧 `poc/src/realtime/feedback/` ode_rollout 范式（部分组件如 rate_limiter 复用，详 codex commit message）
- **记忆**：`feedback_codex_mock_test_blindspot.md` / `feedback_pyfeat_tmpdir_race.md` / `feedback_user_intent_anchoring.md`

---

**签字**：川哥 2026-05-24 凌晨钦定（Q1=A/Q2=A/Q3=A + team 模式最大努力）
**主线**：cc（Claude）
**协同**：codex（gpt-5.3-codex high） · gemini（3-flash-preview）
