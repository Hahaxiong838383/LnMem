# UI v0 设计：Tauri 桌面悬浮 widget + 标注接口

> Sprint 1 输出 / cc 主线设计 / 2026-05-23

## 1. 设计原则

1. **常驻悬浮**：始终在屏幕角落显示，不抢焦点
2. **观察期不打扰**：Sprint 1 仅显示状态，不弹 prompt（Sprint 2 才主动弹）
3. **低 UI 摩擦**：标注通过点击/手势完成，3 秒内
4. **隐私可控**：仅本地 WebSocket 连接，不外发

## 2. Sprint 1 widget 形态（观察期）

### 2.1 默认状态（120 × 120px 圆角悬浮）

```
┌─────────────────┐
│  🤖             │  ← 状态图标（颜色随 intent 变）
│                 │
│   状态: 观察中  │  ← 文字（Sprint 1 不显示 intent，仅 hidden state 摘要）
│                 │
│   τ: 12 分钟    │  ← 短记忆衰减剩余时长
│                 │
│  [我刚才...]    │  ← 手动标注按钮（Sprint 1 唯一交互）
└─────────────────┘
```

### 2.2 状态颜色编码（Sprint 1 即可展示）
- 🟢 绿：LNN hidden state 稳定（变化小）
- 🟡 黄：LNN hidden state 中等变化
- 🔴 红：LNN hidden state 剧烈变化（应该意味着情绪/状态转折）

### 2.3 手动标注（Sprint 1 数据冷启动）

点击 [我刚才...] 展开 popup：
```
┌──────────────────────────────┐
│  过去 5 分钟我的状态是:       │
│                              │
│  [👍 正反馈]                  │  ← 接受/愉悦/想要被关注
│  [👎 负反馈]                  │  ← 拒绝/烦躁/想被打断
│  [😐 无反馈]                  │  ← 中性/专注/不希望被打扰
│                              │
│  备注（可选）:                │
│  [文本框]                    │
│                              │
│  [提交]  [取消]              │
└──────────────────────────────┘
```

## 3. Sprint 2 主动弹窗形态

### 3.1 主动 prompt（置信度 > 0.85 触发）

屏幕角落弹通知（自动 8 秒消失）：
```
┌──────────────────────────────┐
│  🤖 我猜你现在:               │
│     正反馈倾向 (置信度 0.87) │
│     (你在专注且接近答案)      │
│                              │
│  [✓ 确认]                    │
│  [→ 纠正]                    │  ← 点开后展开三选一
│  [×]                         │  ← 关闭+静默 10 分钟
└──────────────────────────────┘
```

### 3.2 防骚扰规则
- 同类别 5 分钟 cooldown（一次正反馈预测后 5 分钟不再弹正反馈）
- 川哥 [×] 后该类别 10 分钟静默
- 主屏幕全屏应用（视频会议/演讲）→ 全静默
- 23:00-07:00 默认静默

## 4. 后端 API schema

### 4.1 标注上报 `POST /labels`
```json
{
  "ts_start": "2026-05-25T10:25:00Z",
  "ts_end": "2026-05-25T10:30:00Z",
  "label": "positive | negative | none",
  "source": "manual | prompt_confirmed | prompt_corrected",
  "ai_prediction": {
    "label": "positive",
    "confidence": 0.87
  },
  "note": "optional"
}
```

### 4.2 实时状态拉取（WebSocket）
连 `ws://localhost:8765/state`，server 推 30Hz：
```json
{
  "ts": "...",
  "tau_state": {"short_tau_decay_remaining_s": 720},
  "intent_pred": {"label": "...", "confidence": 0.87}  // Sprint 2 才有
}
```

### 4.3 学习曲线拉取 `GET /metrics/learning_curve?days=7`
```json
{
  "daily_accuracy": [0.35, 0.42, 0.51, 0.58, 0.62, 0.65, 0.68],
  "feedback_count": [3, 8, 12, 15, 18, 22, 25],
  "tau_drift_norm": [0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.14]
}
```

## 5. 技术选型

| 选项 | Tauri 桌面 widget | Streamlit web UI |
|------|------------------|------------------|
| 启动方式 | 系统级悬浮窗 + dock 图标 | 浏览器 tab |
| 实时性 | 原生 IPC + WebSocket | 仅 WebSocket（可能 reload）|
| 跨平台 | macOS + Linux + Win | 同 |
| 川哥体验 | 24/7 常驻不抢焦点 | 必须开浏览器 tab |
| 开发量 | 中（Rust + 前端）| 小（纯 Python）|
| 用户感知 | 像系统级伴侣 | 像 dashboard |

**cc 推荐**：先 **Streamlit v0**（Sprint 1 一周内能跑），Sprint 2-3 再考虑迁移 Tauri（如果 Streamlit 太重打扰心流）。

## 6. Sprint 1 唯一交付

- Streamlit app `web/ui_v0.py`
- 显示实时 LNN state + τ 衰减
- 手动 [我刚才...] 标注按钮（写到 `~/.lnmem/labels/YYYY-MM-DD.jsonl`）
- 启动方式：`streamlit run web/ui_v0.py --server.port 7860`

## 7. Sprint 2 升级清单（不在 Sprint 1 做）
- 主动 prompt + cooldown 防骚扰
- 三分类置信度展示
- 反馈学习曲线
- Tauri 迁移评估

---

**📝 cc 主线设计** | 2026-05-23
