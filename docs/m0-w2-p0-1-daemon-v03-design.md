# M0 W2 P0-1: daemon V0.3 设计 — user_triggered_mismatch channel + trigger 平衡

> 状态：cc 草稿（待 codex 对抗审 → 川哥签）
> 作者：cc | 2026-05-11 09:00 CST
> 范围：daemon trigger 子系统 + 反馈 schema + CLI 客户端

---

## A. 问题陈述

**根因数据**（PoC Day 9-14 累积，2026-05-04 至 2026-05-11）：

| label | 累积 | 占比 |
|-------|------|------|
| match | 12 | 46% |
| skip | 10 | 38% |
| input | 3 | 12% |
| **mismatch** | **1** | **4%** |
| **总** | **26** | 100% |

**痛点链**：

1. daemon V0.2 trigger 设计**偏 low_vlm_conf 路径**（"系统自己拿不准 → 弹窗问"）：26 次 trigger 中 low_vlm_conf 占 61%
2. 川哥对 low_vlm_conf 类弹窗**主要 skip 不回应**（弹窗提示"皱眉=0.00 笑=1.00 ..."，川哥感知不到自己当下被分类成什么状态，无从对错）
3. brow_burst 类 trigger（阈值 > 0.85）触发**都是高强度皱眉**，川哥认真回应但**几乎都是 match**（"是的我在思考"），少有 mismatch
4. **结果**：mismatch 6 天累积仅 1 条，远低于 [online_trainer.py:45](poc/src/lnn/online_trainer.py:45) `trigger_if_ready` 启动门槛 `min_mismatch_for_fit ≥ 10`
5. **Phase 2 在线学习闭环工程链 100% 实现但永远启动不了** — daemon V0.2 没有让川哥**主动告诉模型"你刚才判错了"**的通道

**核心命题**：现有 trigger 模式假设"模型不确定时问用户" → 但用户对模型确定的错判更有动力纠正；缺这条通道，反馈数据偏向收集"系统知道自己不确定"，不收集"系统不知道自己错了"，正是 mismatch 短缺的根因。

---

## B. 设计目标

1. **新增 user_triggered_mismatch channel**：让川哥**主动**告知系统当前状态判断错误（不等系统问）
2. **trigger 阈值平衡**：减少 low_vlm_conf 类过度触发，多样化 brow_burst 之外的被动 trigger
3. **反馈 schema 向后兼容**：V0.3 是叠加而非替换；V0.2 累积的 26 条反馈仍可用作 baseline
4. **LNN-FIRST P-1 合规**：新 channel 写入相同 JSONL 格式，被同一份 [feedback_to_batch.py](poc/src/lnn/feedback_to_batch.py) 消费，下游 fit_step 无改动
5. **可观测**：新加 trigger 类型分布 metric，写入 daemon `/health` 让川哥实时看分布

---

## C. 设计方案

### C.1 新增 trigger 类型：`user_triggered`

| 属性 | 值 |
|------|---|
| trigger 主体 | 用户主动（不依赖 daemon 信号判断） |
| 写入字段 | `reason: "user_triggered"`，`signal: null`（用户态触发无信号快照） |
| user_response.label | `mismatch`（默认）或 `input`（用户提供正确 state） |
| true_state | 必填（用户输入的真实状态） |
| pred_state | 当前 daemon 推理结果（daemon 写） |
| note | 选填（用户自由文字） |
| event_uuid | 必填（UUID4） |
| session_id | 当前 daemon session |
| frame_id | 当前 frame_id（daemon 最近一帧） |

**示例 JSONL 行**：
```json
{
  "ts": "2026-05-11T09:30:00.000000Z",
  "frame_id": 35000,
  "reason": "user_triggered",
  "signal": null,
  "vlm_label": null,
  "user_response": {
    "label": "mismatch",
    "true_state": "engaged_focused",
    "note": "我在认真写文档不是分心"
  },
  "latency_s": 0.0,
  "session_id": "<current>",
  "event_uuid": "<uuid4>"
}
```

### C.2 trigger 阈值平衡（V0.2 既有 trigger 调整）

| trigger 类型 | V0.2 阈值 | V0.3 阈值 | 调整理由 |
|------------|----------|----------|---------|
| low_vlm_conf | confidence < 0.65 | confidence < 0.5 | 收紧（避免 VLM 中等不确定时也弹窗） |
| brow_burst | brow_furrow > 0.85 ∧ window=10 | brow_furrow > 0.7 ∧ window=15 | 放宽阈值 + 加长窗口（中度皱眉也可触发，但需更稳定） |
| smile_burst | (新增) | smile_intensity > 0.7 ∧ window=15 | 新增对称 trigger（避免单一情绪偏好） |
| pose_held | (新增) | pitch/yaw 偏离 > 30° ∧ window=20 | 新增姿势 trigger（如长时间低头/侧脸） |

**rate-limit / DnD 不变**：现有 `feedback/persistence.py` 限速 + DnD 窗口配置保留。

### C.3 数据接口

#### C.3.1 daemon HTTP 端点（新增）

**POST /trigger_mismatch**

请求 body（JSON）：
```json
{
  "true_state": "engaged_focused",      // required
  "note": "我在写文档",                  // optional
  "label": "mismatch"                   // optional, default="mismatch", enum=["mismatch","input"]
}
```

响应：
```json
{
  "ok": true,
  "event_uuid": "<uuid4>",
  "frame_id": 35000,
  "pred_state": "low_energy"             // daemon 当前推理的 state（供用户对照）
}
```

错误响应（4xx）：
- 400: true_state 不在合法枚举内
- 400: label 不在 {"mismatch", "input"}
- 503: daemon 未推理过任何帧（frame_id 为 0）

#### C.3.2 CLI 客户端（新增）

文件：`scripts/lnmem-tell`（bash 简单 wrapper）

用法：
```bash
lnmem-tell <true_state> [note]
# 例：
lnmem-tell engaged_focused "我在写文档"
lnmem-tell tired
```

实现：curl POST 到 daemon /trigger_mismatch，pretty print 响应。

#### C.3.3 反馈 schema 向后兼容

- 现有 [persistence.py:8](poc/src/realtime/feedback/persistence.py:8) reason 类型增至 3 个：`brow_burst | low_vlm_conf | user_triggered`
- queries_YYYY-MM-DD.jsonl 文件格式不变（新字段 `reason=user_triggered` + `signal=null` 是已支持的 schema 扩展）
- [feedback_to_batch.py](poc/src/lnn/feedback_to_batch.py) 解析 user_triggered 反馈：signal=null 时跳过 fusion vector slicing，但**保留 mismatch label 进入训练批**（关键）

#### C.3.4 trigger 类型分布 metric（新增 /health 字段）

```json
{
  "feedback": {
    "triggered": 203,
    ...
    "trigger_types": {                  // ← 新增
      "brow_burst": 7,
      "low_vlm_conf": 11,
      "smile_burst": 0,
      "pose_held": 0,
      "user_triggered": 0
    }
  }
}
```

---

## D. 实现方案

### D.1 daemon 改动点

| 文件 | 改动 | 行数估 |
|------|------|--------|
| `poc/src/realtime/feedback/trigger.py` | 阈值常量调整 + 新 trigger 类型 (smile_burst / pose_held) | +60 / 修 10 |
| `poc/src/realtime/feedback/persistence.py` | reason 枚举扩展 + trigger_types metric 计数 | +20 |
| `poc/src/realtime/daemon_main.py` | HTTP server 加 POST /trigger_mismatch handler | +50 |
| `poc/src/realtime/daemon_main.py` | /health 输出 trigger_types | +5 |
| `poc/src/realtime/feedback/engine.py` | (无改动) | 0 |
| `poc/src/lnn/feedback_to_batch.py` | user_triggered 反馈（signal=null）的解析路径 | +20 |
| `scripts/lnmem-tell` | 新建 CLI wrapper | +30 |

**总估**：~200 行新增 + 10 行修改。

### D.2 新文件清单

```
poc/src/realtime/feedback/test_trigger_user_triggered.py   # 单测（新）
poc/src/realtime/test_daemon_trigger_mismatch.py            # 集成测（新）
scripts/lnmem-tell                                          # CLI（新）
docs/m0-w2-p0-1-daemon-v03-design.md                        # 本文档
```

### D.3 测试方案（10 case 覆盖率目标）

**单测 test_trigger_user_triggered.py**（4 case）：
1. POST /trigger_mismatch 正常 → 200 + 写 JSONL + frame_id 来自最近一帧
2. POST 缺 true_state → 400
3. POST true_state 不合法枚举 → 400
4. POST label=input → JSONL 写入 label=input 不是 mismatch

**集成测 test_daemon_trigger_mismatch.py**（3 case）：
5. CLI `lnmem-tell engaged_focused` → daemon → JSONL 出现 → reason=user_triggered
6. user_triggered 反馈被 [feedback_to_batch.py](poc/src/lnn/feedback_to_batch.py) 解析为 mismatch 进入训练批
7. trigger_if_ready 在 user_triggered ≥ 10 mismatch 时返 True

**回归 test_trigger_v02.py 既有**（3 case 保留）：
8. brow_burst 阈值 0.7 新值不破现有 7 条 brow_burst 反馈识别
9. low_vlm_conf 阈值 0.5 新值过滤掉 V0.2 偏激进的中等不确定弹窗
10. trigger_types metric 计数与历史 26 反馈一致

---

## E. 双门验证（实施后）

### E.1 Verify Gate（目标验证）

- [ ] 新加 10 个 test case 全 PASS
- [ ] 川哥试用 1 周累积 user_triggered mismatch ≥ 5（占新增反馈 ≥ 30%）
- [ ] trigger_types metric 显示分布更均匀（任一类型占比 < 70%）
- [ ] feedback_to_batch dry-run sanity 通过（user_triggered 反馈进训练批）

### E.2 Guard Gate（回归防护）

- [ ] V0.2 既有 373 测试全 PASS
- [ ] V0.2 累积 26 条反馈 schema 仍可被 feedback_to_batch 解析
- [ ] daemon `/health` 既有字段全保留
- [ ] watchdog W1-P2-1 修复后的脚本不受影响

---

## F. 时间盒

| 阶段 | 估时 | 负责 |
|------|------|------|
| 设计文档（本文） | 1h | cc ✅ |
| codex 对抗审 | 30-60min（async） | codex (gpt-5.3-codex high) |
| 川哥签字 | 不定 | 川哥 |
| 实现（daemon + CLI + test） | 4-6h | cc 主线 |
| Verify Gate 测试 | 1h | cc |
| Guard Gate 回归 | 30min | cc |
| 川哥 1 周试用 | 7 天（async 等数据） | 川哥日常使用 |
| 验收（mismatch 累积 + metric 检查） | 1h | cc |

**M0 W2 P0-1 总估**：cc 主线 7-9h（含设计 + 实现 + 测试 + 验收），川哥试用 7 天累积数据。

---

## G. 不做的事（YAGNI）

1. ❌ 不做桌面状态栏小工具（保 CLI 即可，桌面 UI 推迟到 M1）
2. ❌ 不做全局快捷键监听（macOS 全局快捷键需 Accessibility 授权，过度复杂）
3. ❌ 不做 user_triggered 反馈的 reward 加权调整（默认 +1，与既有 match 同等；后续 P0-3 vboss_self 微调时再调）
4. ❌ 不改 reward_map（保持 [online_contracts.py](poc/src/lnn/online_contracts.py) REWARD_MAP 现状）
5. ❌ 不接入 daemon V0.3 自学习闭环（那是 P0-3 范围；P0-1 只解决反馈通道）

---

## H. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| trigger 阈值放宽后过敏（弹窗 > 30/h） | 中 | 用户体验下降 | rate-limit 防线在；首日川哥手动监控弹窗频率，过敏立即回滚 |
| CLI 客户端使用门槛高（川哥不习惯命令行） | 中 | mismatch 累积仍慢 | 提供 1-key alias / 后续做桌面快捷键 |
| user_triggered 反馈的 frame_id 与 daemon 最近一帧错位（用户思考 → 输入 → 提交期间帧已往前） | 低 | 训练用错 fusion vector | frame_id 取**daemon 接收 POST 时**最近一帧，记录时间差 < 5s 才入训练批 |
| smile_burst / pose_held 新 trigger 类型 V0.3 首次见，调参不对 | 中 | 弹窗分布反而更偏 | 首周 metric 监控，> 70% 单类型占比 → 阈值微调 |

---

## I. 关联文档与依赖

- 上游设计：[`docs/poc-final-report.md`](poc-final-report.md) §C.3（PoC 反馈类型分布异常发现）
- 上游设计：[`docs/m0-w2-tasks.md`](m0-w2-tasks.md) P0-1 任务定义
- 下游依赖：M0 W2 P0-2（feedback v2 schema 加固）依赖本 P0-1 reason 枚举扩展
- 下游依赖：M0 W2 P0-3（vboss_self 微调）依赖本 P0-1 mismatch 累积 ≥ 10
- 已有实现：[`online_trainer.py`](../poc/src/lnn/online_trainer.py) / [`online_adapter.py`](../poc/src/lnn/online_adapter.py) / [`feedback_to_batch.py`](../poc/src/lnn/feedback_to_batch.py)（PoC 已完成）

---

## J. codex 对抗审 prompt（cc 派发用）

```
你的角色：对抗审查者。审查 LnMem M0 W2 P0-1 daemon V0.3 设计文档。
目标：找出设计漏洞 / 反例 / 实现陷阱 / 可证伪假设，不做客气评价。

文档位置：docs/m0-w2-p0-1-daemon-v03-design.md (相对仓根)
背景：LnMem PoC 14 天已完成，自学习闭环工程链已实现但 mismatch 反馈结构性短缺导致 Phase 2
门槛不可达。本设计为解此痛点，新增用户主动触发 mismatch 通道。

审查维度（必查）：
1. 设计目标是否真的解决问题陈述？是否存在隐含假设？
2. user_triggered 反馈的 frame_id 与 daemon 最近一帧对齐策略是否健壮？
3. trigger 阈值调整（brow_burst 0.85 → 0.7 + window 10 → 15）是否会过敏？历史
   PoC 数据 7 条 brow_burst 都是 brow≥0.9 全 match，放宽后会引入多少噪音？
4. user_triggered 反馈被加入训练批的 reward 默认 +1，是否会让模型过拟合到用户主观
   判断（用户也可能记错）？
5. feedback_to_batch.py 对 signal=null 反馈的处理路径是否会破坏现有 (sid, fid) 复合键
   slice_window 逻辑？
6. /health trigger_types metric 新加字段是否影响现有 client（如 watchdog）解析？
7. CLI 客户端 lnmem-tell 缺权限校验（任何能 curl 127.0.0.1:9876 的进程都可写入反馈），
   是否风险？
8. LNN-FIRST P-1 合规：新 trigger 类型有无破坏 LTC ODE forward 路径假设？
9. 测试方案覆盖率：是否有边界 case 漏掉？
10. 时间盒估算 4-6h 实现是否乐观？

输出格式：
- 每个 finding 标 CRITICAL / HIGH / MEDIUM / LOW
- 给出具体反例 / 代码片段证据 / 修复建议
- 最后一段 ## Summary 给 PASS / PASS-WITH-CHANGES / BLOCK 决议
```

---

## K. 签字栏

- cc 设计完成：2026-05-11 09:00 CST ✅
- codex 对抗审：⏳ 待派发
- 川哥签字：⏳ 待审
- 实现启动：⏳ 待签后
