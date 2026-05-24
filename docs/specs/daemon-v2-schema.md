---
version: v0.1
audience: cc / codex / 工程师
purpose: 定义 daemon v2 的 30Hz 多模态 JSONL schema 与兼容策略
status: draft
last_updated: 2026-05-24
---

# Daemon v2 Schema (30Hz Multimodal)

| RULE-ID | 规则 | 要求 |
|---|---|---|
| DAEMON-V2-001 | 采样频率 | `target_fps = 30.0` |
| DAEMON-V2-002 | AU 向量 | `au` 必须 20 维，顺序与 `PYFEAT_AU_COLUMNS` 一致 |
| DAEMON-V2-003 | 头部姿态 | `head_pose = {yaw_deg, pitch_deg, roll_deg}` |
| DAEMON-V2-004 | 屏幕活动 | `screen_activity = {active_app, key_rate_per_s, mouse_rate_per_s}` |
| DAEMON-V2-005 | 兼容策略 | 缺失 `head_pose/screen_activity` 时下游填 0，不中断训练/推理 |

**Why**：先把数据契约固定，才能确保 30Hz 训练与实时推理使用同一语义。

## JSONL Row Schema

```json
{
  "ts": "2026-05-25T10:30:00.033Z",
  "frame_id": 12345,
  "session_id": "uuid",
  "face_detected": true,
  "calibrated_au": { "AU01": 0.1, "...": 0.2, "AU43": 0.0 },
  "raw_au": { "AU01": 0.1, "...": 0.2, "AU43": 0.0 },
  "head_pose": {
    "yaw_deg": 1.2,
    "pitch_deg": -3.4,
    "roll_deg": 0.8
  },
  "screen_activity": {
    "active_app": "Cursor",
    "key_rate_per_s": 2.1,
    "mouse_rate_per_s": 1.3
  }
}
```

**Why**：`head_pose` 与 `screen_activity` 并入同一帧事件，保证 ODE 更新使用同一时间切片。

## Cache Projection (for `daemon_dataset.py`)

- `au_<session_short>`: `(T, 20)` float32
- `frame_<session_short>`: `(T,)` int64
- `headpose_<session_short>`: `(T, 3)` float32
- `screen_<session_short>`: `(T, 3)` float32
  - `screen[:,0]`: active_app presence flag (`1.0` non-empty else `0.0`)
  - `screen[:,1]`: key_rate_per_s
  - `screen[:,2]`: mouse_rate_per_s

**Why**：缓存层统一 dense tensor，训练/推理避免在热路径反复解析 JSON。
