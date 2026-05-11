---
name: LnMem PoC Day 13 收尾报告
description: Day 13 交付汇总 / 反馈分布异常分析 / Day 14 计划 / PoC 时间盒重评估
type: end-of-day
created: 2026-05-10
phase: PoC Day 13 of 14 (剩 1 天)
status: closed_with_partial_dod_engineering_full
---

# LnMem PoC Day 13 收尾报告

> **2026-05-10 17:15 CST 闭环**。Day 13 = PoC 倒数第二天。
> **工程层面 Day 13 全部 100% 交付**（feedback_to_batch 工程链 + dry-run + codex 三道门 + 7 finding 全修）；
> **数据采集 1/3 部分达标**：反馈 total 达门槛但 mismatch 严重不足，real time 距 15h 仍缺 2.7h。

---

## A. 本日交付清单（4 轮 Loop / 4 commit）

### A.1 Loop 4（Day 12 末 15:10 · commit `f887235`）— Day 12 收尾报告
（跨 Day 12-13 交接，作为 Day 13 起点）

### A.2 Loop 5（15:55 · commit `b4925e4`）— P0-2 feedback_to_batch 工程链 + 真实数据 dry-run
**新增**：`signals_loader.py`(180) + `feedback_to_batch.py`(200) + 26 TDD
**关键里程碑**：**PoC 端到端 dry-run sanity ✅** — 真实 daemon 数据从 fit_step → tau_drift 反传通跑
**实测证据**：vboss_self idx=30 tau_drift L2=0.000004 唯一动 / vboss_001-030 L2=0
**Gate 2**：354/354 全量回归 PASS

### A.3 Loop 6（16:35 · commit `3c415a4`）— Codex 三道门中间门 + 7 finding 全修
**Codex 对抗审**：gpt-5.3-codex/high session 019e10fe-1ba2 后台 5min → 7 finding（**2 阻断 F1 + 2 高 F2 + 2 中 F3 + 1 低 F4**）

| ID | 修复 | 实测影响 |
|----|------|---------|
| **F1-01** 跨天 frame_id silent overwrite | `build_session_frame_index` (sid, fid) 复合键 + V0.1 fallback | dry-run batch **4→3**：5/9 frame_id=0 跨 session 错配样本被丢，**修复立竿见影** |
| **F1-02** -1 占位 gather 错训 | `compute_reward_loss` hard assert + OnlineAdapterError 提示 | 防 silent silent error |
| F2-03 match argmax noise | `confidence_threshold=0.30` 默认门槛 + 低置信降级 reward=0 | dry-run 3 match 全过 0.30 阈值 |
| F2-04 missing 语义污染 | 3 字段 RuntimeWarning（PoC 妥协，M0 W2 missing mask） | — |
| F3-05 窗口连续性 | session boundary + 首帧 frame_id 校验 | — |
| F3-06 frozen tensor | `.clone()` 防御 in-place | 测试期望同步更新 |
| F4-07 true_state alias | `TRUE_STATE_ALIASES` + `_normalize_true_state` | 覆盖 hyphen/lowercase/同义词 |

**Gate 2**：373/373 全量回归 PASS（无退化，354→373 = +19）
**修复后 dry-run**：loss **-1.73→0.147** 数值合理 / batch 4→3 错配丢

### A.4 Loop 7（17:15 · 本报告）— Day 13 收尾决策

---

## B. Day 13 末 DoD 验收

| 项 | 目标 | 当前 | 进度 | 状态 |
|----|------|------|------|------|
| feedback_to_batch 工程链 | 就位 | ✅ commit b4925e4 | 100% | ✅ |
| 真实数据端到端 dry-run | 通跑 | ✅ vboss_self drift L2 唯一动 | 100% | ✅ |
| Codex 对抗审三道门 | 通过 | ✅ 7 finding 全修 | 100% | ✅ |
| Gate 2 全量回归 | 无退化 | ✅ 373/373 | 100% | ✅ |
| docs/day13-progress.md | 时序日志 | ✅ §A-§G + §C.6 codex review | 100% | ✅ |
| RECENT_EVENTS + task-log | 追加 | ✅ 3 轮记录（Loop 5/6/7） | 100% | ✅ |
| daemon 稳定 ≥6h | 0 ERROR | ⏳ 当前 5.13h | 86% | ⏳ |
| 反馈累积 ≥10 total | 最低门槛 | ✅ 10 条 | **100%** | ✅ |
| 反馈 ≥10 mismatch | Phase 2 实战触发 | ⏳ **当前仅 1 条** | **10%** | ❌ |
| 真实数据 ≥15h（Day 13 末）| 进 Day 14 门槛 | ⏳ 当前 12.3h | 82% | ⏳ |

**结论**：
- 工程层面 6/6 全交付（**100% 完美**）
- 数据采集 4 项中 1 项达标（反馈 total）/ 3 项 unmet：daemon 86% / mismatch 10% / signals 82%
- **不可妥协约束 #5「真实数据 <15h 不进 Day 13」**：当前 12.3h < 15h，**严格意义上 Day 13 完结时数据不达标**

---

## C. 反馈累积分布异常分析（Day 13 关键发现）

### C.1 完整反馈表（10 条总）

| # | ts | reason | label | true_state | session_id | 备注 |
|---|----|----|------|----|----|----|
| 1 | 5/9 12:56 | low_vlm_conf | mismatch | **distracted** | None (V0.1) | 光线极暗 conf=0.05 |
| 2 | 5/9 13:16 | low_vlm_conf | input | — | None (V0.1) | 川哥："我不在，检测错了吧" |
| 3 | 5/10 04:05 | brow_burst | match | — | 41f0... | brow=1.0 |
| 4 | 5/10 04:32 | low_vlm_conf | skip | — | 41f0... | FB-09 v2 漏（已修） |
| 5 | 5/10 04:54 | brow_burst | match | — | 41f0... | brow=0.916 + AU4+AU7 疲倦专注 |
| 6 | 5/10 14:16 | low_vlm_conf | skip | — | 41f0... | FB-09 v3 漏（已修） |
| 7 | 5/10 15:26 | brow_burst | match | — | 41f0... | brow=1.0 |
| 8 | 5/10 16:03 | low_vlm_conf | skip | — | 41f0... | conf=0.1 临界（FB-09 不漏，川哥合理 skip）|
| 9 | 5/10 16:34 | low_vlm_conf | skip | — | 41f0... | 新增 |
| 10 | 5/10 16:52 | low_vlm_conf | skip | — | 41f0... | 新增 |

### C.2 关键观察

**反馈类型分布**：
- match × 3 (30%)
- mismatch × 1 (10%)
- skip × 5 (50%) ← **过半**
- input × 1 (10%)

**有效训练信号**：3 match + 1 mismatch = **4 条**（占总样本 40%）

**问题诊断**：
1. **skip 占比 50% 过高**：意味着 daemon trigger 弹窗中一半川哥认为"非典型时刻不该问"。说明 trigger 阈值（brow_burst 0.85+10帧 / low_vlm_conf 0.65）仍偏松。
2. **mismatch 仅 1 条**：daemon 5/10 一整天 0 mismatch（仅 5/9 那条已知漏过的"光线极暗 VLM 自报无脸"伪信号），说明 daemon 推断与川哥真实状态一致性较高（好信号）但**对 Phase 2 训练意味着监督信号不足**（坏信号）。
3. **brow_burst 触发的全是 match**（3/3）：川哥皱眉时确实在专注/思考，模型预测正确——但这种"自我验证"式 reward 仅强化已学好的状态，不产生纠偏信号。
4. **5 条 skip 全来自 low_vlm_conf**（VLM confidence 0.0-0.1）：daemon 在不确定时主动问，但川哥没法标 → 信号浪费。

### C.3 Phase 2 启动的真正卡点

trigger_if_ready 双门：
- ✅ total ≥ 10 = 10/10
- ❌ mismatch ≥ 10 = 1/10

**仅 mismatch 不足**。这不是反馈量问题，是**反馈类型分布**问题：
- 模型当前预测准确率较高（mismatch 罕见）
- daemon 的 trigger 策略偏向"模型自己拿不准"（low_vlm_conf）而非"模型错预测"（user 主动纠正）

**M0 W2 改进方向**：
- daemon 加 `user_triggered_mismatch` channel：川哥主动按快捷键报告"模型搞错了"
- low_vlm_conf trigger 阈值上调（0.65→0.50），减少 skip 浪费

---

## D. 真实数据 25h 硬门槛进度

| 来源 | 帧数 | 时长 (≈1 fps) |
|------|------|--------------|
| signals_2026-05-09.jsonl | 27860 | **7.74h** |
| signals_2026-05-10.jsonl | 16378 | **4.55h** |
| **总计** | **44238** | **12.29h** |

**距 Day 13 末门槛（15h）**：**缺 2.71h**
**距 Day 14 末门槛（25h）**：**缺 12.71h**

### D.1 Day 14 数据采集预估

假设川哥 Day 14 daemon 使用：
- **4-6h（保守）**：14.3-16.3h 总计 → **不达 25h**
- **8-10h（积极）**：18.3-20.3h 总计 → **不达 25h**
- **12-14h（极限）**：22.3-24.3h 总计 → **接近但仍不达**

**结论**：**25h 硬门槛不可能达成**（Day 14 单天剩余时段不足）。

---

## E. PoC 时间盒重评估（建议给川哥）

### E.1 当前事实

- 工程层面 Day 12 + Day 13 100% 交付（Phase 2/3 + vboss_self + Codex × 2 三道门 + dry-run）
- 反馈累积 mismatch 严重不足（1/10），daemon 触发策略需 M0 W2 调整
- 25h 真实数据不可能达成，最多 ~22h 极限

### E.2 三个候选方向（cc 立场 + 建议）

**方案 A：严格按 PoC 不可妥协约束 → 暂停 PoC，回到 daemon V0.3 改造**
- 立场：约束第一，工程纪律
- 行动：Day 14 不跑 Phase 2，改造 daemon V0.3（user_triggered_mismatch + 触发阈值上调）+ 延长 daemon 使用至 ≥25h（但跨 1-3 天）
- 代价：PoC 时间盒延长，但数据严谨

**方案 B：放宽 25h 硬门槛 → Day 14 启 Phase 2 with 现有数据**
- 立场：工程演示优先，时间盒优先
- 行动：Day 14 用 4 有效样本（合成扩样禁止）启首次 fit_step 实战 + Demo 录屏 + 收尾报告标注"数据不足"
- 代价：PoC 科学结论标 partial，但完成时间盒

**方案 C（cc 推荐）：dry-run 即 PoC 实质交付 + 接受数据采集是 M0 W2 主任务**
- 立场：**PoC 工程闭环已是核心价值**，反馈累积是产品迭代而非 PoC 验证
- 行动：Day 14 = (a) 写 PoC 收尾报告（核心强调工程链 + dry-run + codex × 2 三道门 + 修复立竿见影）+ (b) Demo 录屏 + (c) M0 W2 任务清单（含 daemon V0.3 + 反馈类型平衡）
- 代价：M0 W2 必须接收数据采集任务（已计入路线图）

**cc 倾向 C**：理由是 PoC 的"证伪假设"价值在于工程链——dry-run 已证明"反馈 → reward → tau_drift 真触达 LTC ODE"，这是 LNN-FIRST 范式的核心证据。强行用 4 样本跑 fit_step 反而引入"低样本量结果不可信"的噪声。

---

## F. Day 14 计划（cc 推荐方案 C）

### F.1 P0 任务

#### P0-1 PoC 收尾报告 docs/poc-final-report.md
- 14 天产出汇总（Day 0-13）
- 关键技术决策（CfC 主路径 / vboss_self informed init / 三道门工程纪律）
- 工程链端到端证据（commit hashes + tests counts + dry-run logs）
- 与 v0.3.2 product-definition 的差距 + M0 W2 路线图

#### P0-2 Demo 录屏（5-10min）
- 摄像头实时预览 + daemon /health 浮窗
- 弹窗反馈实拍（让川哥真实做几个表情触发）
- ckpt 加载 + dry-run + tau_drift 变化动画
- 三道门工程纪律展示（cc → codex review report 截图）

#### P0-3 M0 W2 任务清单 docs/m0-w2-tasks.md
- daemon V0.3：user_triggered_mismatch channel + trigger 阈值平衡
- feedback_to_batch v2：missing mask + (sid, fid) 强制（去 V0.1 fallback）
- vboss_self 真实反馈微调（M0 W2 攒够 ≥10 mismatch）
- input free-form text NLU（提取隐含 reward 信号）

### F.2 P1 任务（视精力）

- daemon 持续运行至凌晨累积 reward（M0 W2 启动数据基础）
- mycc memory 沉淀本周教训（codex 三道门实战价值 / FB-09 v1→v2→v3 演进 / dry-run 即 PoC 价值）

### F.3 不在 Day 14 做（避免范围爆炸）

- 不跑 Phase 2 实战微调（4 样本太少）
- 不修 F2-04 missing mask（M0 W2 主任务）
- 不开 daemon V0.3 改造（M0 W2 主任务）

---

## G. 风险与待解项

### G.1 已知风险

1. **mismatch 反馈结构性短缺**（C.3 分析）：daemon V0.2 trigger 策略偏 low_vlm_conf 不偏 user 主动纠正
2. **25h 硬门槛不可达**（D.1）：Day 14 单天最多累积 ~9-10h 增量
3. **5 条 skip 信号浪费**（50%）：弹窗触发但川哥无法标
4. **input free-form 1 条 + mismatch 1 条**：少数派类型未释放价值

### G.2 待向川哥决策项

- [ ] 选方案 A / B / C
- [ ] M0 W2 路线图同意吗（daemon V0.3 + missing mask + 真实反馈微调）
- [ ] Demo 录屏是否需要川哥配合（5-10min 拍摄）
- [ ] PoC 时间盒延长是否接受（如选 A）

---

## H. Loop 7 daemon /health 快照 #11（17:11 CST）

```json
{
  "status": "ok",
  "uptime_s": 18476.933,    // 5.13h
  "frames_processed": 16378,
  "last_frame_ts": "2026-05-10T09:11:10Z",
  "feedback": {
    "observed": 16378,
    "triggered": 56,         // +7 vs Loop 6
    "rate_blocked": 48,      // +5
    "dnd_blocked": 0,
    "queried": 8,            // +2 川哥进帐
    "responses": 8           // 同步
  }
}
```

PID 15934 RSS 1.10GB（稳定），距 6h DoD 还需 0.87h（可达，川哥若再用 1h）。

---

## I. cc 自评

**Day 13 cc 主线表现**：
- ✅ 4 轮 Loop 自驱 / 4 commit / 工程链 100% 交付
- ✅ Codex 三道门主动派审（vs Day 12 cc 主动）+ 7 finding 全修
- ✅ 修复后 dry-run 实证 F1-01 修复价值（错配样本立竿见影丢弃）
- ✅ 所有 commit 含详细原因 + Refs + 双门验证证据
- ✅ docs/day13-progress.md 7 段完整 + 决策记录 F.1-F.3
- ⚠️ mismatch 反馈短缺**未及早识别**——直到 Loop 7 才发现 daemon trigger 策略问题
- ⚠️ PoC 时间盒重评估**应更早提**（Day 12 末或 Day 13 中）

**改进**：下次 PoC（M1）开始时即设 mid-point 数据健康度评审节点（Day N/2），而非等 Day N-1 才发现结构性问题。
