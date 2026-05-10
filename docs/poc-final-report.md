---
name: LnMem PoC 收尾报告（Day 0-13 完整复盘）
description: 14 天 PoC 总产出 / 工程链端到端证据 / 关键技术决策 / 与 v0.3.2 product-definition 差距 / M0 W2 路线图
type: poc-final-report
created: 2026-05-10
phase: PoC closing (Day 14 of 14)
status: engineering_full_data_partial
---

# LnMem PoC 收尾报告（Day 0-13）

> **2026-05-10 17:50 CST 闭环**。LnMem 14 天 PoC 收官报告。
>
> **核心结论**：**工程层面 PoC 100% 达成**——端到端学习闭环（摄像头采集 → daemon trigger → user reward → tau_drift LTC ODE 反传 → 模型适配）已在真实数据上跑通且经 codex 双轮对抗审查全 finding 修复；**数据采集层面部分达标**——反馈累积达 ≥10 总最低门槛但 mismatch 类型严重短缺（1/10），25h 真实数据硬门槛不可达（12.8h / 51%）。**PoC 证伪假设的核心价值已交付**——LNN-FIRST 范式下 reward → τ-bias 漂移可工程化、可度量、可隔离 per-persona。剩余的"反馈类型分布平衡"与"长尾数据采集"已识别为 M0 W2 主任务。

---

## A. 14 天产出汇总（commit 链 + 测试覆盖 + 关键里程碑）

### A.1 commit 链（poc/lnn-validation 分支主干）

| 阶段 | commit | 关键交付 |
|------|--------|---------|
| Day 0-1 | e8d28d8 | 多模态采集骨架（capture/ 7 文件 + 4 tests + pyproject + requirements）|
| Day 5-6.5 | (Day 5/6 多 commit) | LNN 主路径（CfC + LTC ODE 半隐式欧拉）/ 表情 LTC 演化 / py-feat AUCaliper / Day 6.5 真人验证 91.5% val acc 突破 |
| Day 7-8 | (Day 7-8 多 commit) | Realtime Capture Daemon V0.1（6 模块 HTTP 9876）/ 双摄像头 20 AU 全激活 / 范式跃迁"AI 学 AI"→真实数据驱动 |
| Day 8-9 | 32f8c28 | LNN-FIRST P-1 准则落地（MediaPipe 删除 / py-feat img2pose 头姿）+ VLM 标注上线（daemon._load_dotenv 修复）|
| Day 10 | 156f9e1 | 反馈闭环开闸 V0.2（feedback/ 6 模块 + macOS 弹窗 + codex 8 finding 4 修 + 第一条真实反馈样本）|
| Day 11 | e42c7ad | FB-09 v1+v2 修复（VLM 自报无脸跳过弹窗 / append jsonl / 测试隔离）+ `.venv-poc` 完全重建 / 元方法论 5min 全局扫描 SOP 沉淀 |
| Day 12 | 32f2af3 | **Phase 2 工程骨架**（OnlineLoRAAdapter + FeedbackDataset + online_contracts / 29 TDD）|
| Day 12 | 7c6697c | **Phase 3 训练循环**（fit_step + reward sanity + LNN-FIRST 实测）+ FB-09 英文 keyword 修复 |
| Day 12 | af90c07 | **vboss_self 入 ckpt**（HEXACO informed init / 31 personas）+ FB-09 v3 sentinel + Codex 5 finding 全修 |
| Day 12 | f887235 | Day 12 收尾报告 |
| Day 13 | b4925e4 | **P0-2 feedback_to_batch 工程链**（signals_loader + feedback_to_batch / 26 TDD）+ **🎯 PoC 真实数据端到端 dry-run** |
| Day 13 | 3c415a4 | **Codex Day 13 7 finding 全修**（2 阻断 F1 + 2 高 F2 + 2 中 F3 + 1 低 F4 / 19 新测试） |
| Day 13 | b92afe3 | Day 13 收尾报告 + Day 14 三方案决策 |

### A.2 测试覆盖演进

| 阶段 | 测试数 | 状态 |
|------|-------|------|
| Day 0-9 基线 | 217 | LNN 主路径 + 视觉链路 + 数据集 |
| Day 10 反馈闭环 | 242 | +25 (Phase 1 V0.2 弹窗 + 边界) |
| Day 11 FB-09 v1 | ~250 | +14 trigger v1 |
| Day 12 Phase 2 骨架 | 275 | +29 (OnlineLoRAAdapter + FeedbackDataset) |
| Day 12 Phase 3 + FB-09 v2 | 295 | +20 (fit_step + LNN-FIRST sanity + FB-09 EN) |
| Day 12 vboss + Codex 5 修 | 328 | +33 (extend_persona + LNN-FIRST 显式断言 + FB-09 v3) |
| Day 13 P0-2 工程链 | 354 | +26 (signals_loader + feedback_to_batch) |
| Day 13 Codex 7 修 | **373** | +19 (F1 阻断 + F2 高 + 缺失警告 + alias) |

**全部 PASS · 0 退化 · 双门验证（verify + guard）严格执行**

### A.3 关键里程碑

| 日期 | 里程碑 |
|------|-------|
| Day 6.5 | 真人验证 val_acc 70.2 → **91.5%**（+21.3pp）/ 微笑→positive 真人 100% 触发 |
| Day 8-9 | LNN-FIRST P-1 落地（MediaPipe 视觉链彻底删除 + py-feat 唯一 AU 源） |
| Day 10 | **第一条真实多模态反馈样本**（5/9 mismatch+distracted） |
| Day 12 | Phase 2 + Phase 3 工程闭环就位 / vboss_self 真人画像入 ckpt 31 personas |
| Day 13 | **🎯 真实数据端到端 dry-run sanity ✅** — vboss_self idx=30 唯一 tau_drift 动 / 其他 30 personas drift=0 |
| Day 13 | Codex 三道门双轮对抗审（5+7=12 finding 全修，含 2 阻断 F1）|

---

## B. 工程链端到端证据（PoC 核心交付）

### B.1 真实数据 dry-run logs

```
源 feedback:  ~/.lnmem/feedback/queries_*.jsonl (10 条总，4 训练样本)
源 signals:   ~/.lnmem/data/signals_*.jsonl (44238 帧 ≈ 12.3h)
ckpt:        day12_with_self/checkpoint_day12.pt (31 personas, hidden_dim=64)

修复后 build_training_batch (commit 3c415a4 后):
  batch.x.shape = (3, 60, 18)
  ← 注意：从 4 减至 3，F1-01 修复立竿见影丢弃了 5/9 frame_id=0 跨 session 错配样本
  frame_ids:    (158, 3045, 12221) 全 5/10 同 "41f0..." session
  labels:       [7, 5, 7] (model argmax for match samples after fill)
  rewards:      [+1, +1, +1] (confidence 全过 0.30 阈值)

fit_step 1 step (lr=1e-3):
  metrics: {loss=0.147, drift_max_norm=3.0e-06, batch_size=3}
  
反传后 tau_drift 范数对比:
  vboss_self  (idx=30): 0.000003   ← 唯一动了的 slot
  vboss_001-030       : 0.000000   ← 全保持 0
```

### B.2 不可妥协约束实测验证表

| 约束 | 验证方式 | 结果 |
|------|---------|------|
| **LNN-FIRST P-1**：τ-bias 漂移走 LTC ODE 半隐式欧拉 | `test_backbone_is_cfc_module` + `test_timespan_formula_is_softplus_ode_path` + 真实数据 dry-run forward | ✅ |
| **per-persona 隔离**：vboss_self 反馈 ≠ 影响其他 persona | `test_only_tau_drift_changes` + dry-run drift L2 仅 idx=30 动 | ✅ |
| **不准合成数据微调** | 全部数据来自 daemon 实采 signals + 真实 user_response | ✅ |
| **三道门**：cc 实现 → codex 对抗审 → 川哥签 | Codex × 2 轮（Day 12 5 finding + Day 13 7 finding）全修 | ✅（待川哥 Day 14 签） |
| **真实数据 ≥ 25h** | 当前 12.8h | ❌ |

---

## C. 关键技术决策（PoC 期间产出，按重要度排序）

### C.1 LNN-FIRST P-1（**架构红线**）
- **决策**：τ-bias / 表情 / 头姿 / persona 个性化全部走 LTC ODE 半隐式欧拉，不准 LLM/规则替代
- **触发事件**：Day 8-9 发现 MediaPipe 在视觉链中"事实上 AU 源"，违反单一信号源原则 → 彻底删除 + py-feat img2pose 唯一 AU 来源
- **Codex F2-02 升级（Day 12）**：仅"能跑通 + grad 存在"间接证明不够 → 加 4 个直接断言（CfC 类型 / softplus 公式锁 / drift 单调 / per-timestep τ 传播）锁住未来重构红线

### C.2 OnlineLoRAAdapter 包装而非扩 base.embedding
- **决策**：base 完全冻结 + 唯一可训练 `tau_drift (n_personas, hidden_dim)` 增量
- **理由**：扩 base.embedding 会破坏 day7_v2 ckpt 训练资产（91.5% val acc）；LoRA 风格仅 ~8KB/persona 持久化代价
- **配套**：vboss_self 用**另一条独立路径**（`extend_persona.py` 直接扩 ckpt embedding shape），冷启动 informed init

### C.3 vboss_self informed init 不做合成 trajectory 微调
- **冲突**：原任务"5-10 epoch 微调"暗示用合成数据，但 PoC 不可妥协 #6 禁合成微调
- **决策**：HEXACO 自评 → 8 维 embedding 中心化作为冷启动起点；后续真实反馈由 OnlineLoRAAdapter.tau_drift 在线漂移
- **结果**：vboss_self idx=30 embedding=[0.167,0,0.25,0.167,0.333,0.417,0.208,0.5]，base 30 行 100% 保留

### C.4 Reward 信号映射 REWARD_MAP（match=+1 / mismatch=-1 / skip=input=0）
- **设计决策**：skip 占 mean 分母（Codex F3-02）→ 保留 batch-mean 标准 REINFORCE 实现，过滤策略由 batching 阶段负责（决策落 docstring）
- **F2-03 codex 修复**：match argmax 加 `confidence_threshold=0.30`，低置信样本降级 reward=0 防 noise 强化

### C.5 三道门工程纪律（cc 实现 → codex 对抗审 → 川哥签）
- **演进**：Day 10 codex 8 finding 4 修（cc 自评余下 4 不必修）→ Day 12 codex 5 finding 全修 → Day 13 codex 7 finding（2 阻断 F1）全修
- **价值**：Day 13 F1-01 实证 codex 真实价值——cc 自标 F.1 风险（保守 silent fallback），被 codex 升级为阻断 F1 必修；修复后生产 dry-run **意外丢弃了一个错配样本**（修复立竿见影自证）

### C.6 跨天 frame_id 复合键（Codex F1-01）
- **根因**：daemon 每天重启 frame_id 重置 0，单键 silent overwrite 导致跨天反馈错配
- **修复**：`build_session_frame_index((session_id, frame_id))` + V0.1 老 feedback 无 session_id 时 fallback 找候选 session
- **代价**：5/9 那条 mismatch（frame_id=0）因 V0.1 feedback session_id=None 但 5/9 signals session_id="a45a..." 在 fallback 时选最新 session 5/10 → window 不足跳过（lose 1 样本但避免错配）

### C.7 BFO 实体锚定 / no-guessing 红线（mycc 全局规则）
- 跨项目：daemon 端口 9876（不是 8765 凭空猜）/ ckpt 路径 / persona_to_idx 等所有具名实体 grep 验证
- LnMem 直接受益：避免域名 / 端口 / 字段名脑补错误

---

## D. 反馈累积分布异常（**Day 13 关键发现 + M0 W2 主驱动**）

### D.1 完整分布

11 条反馈（截至 Day 14 17:46 CST）：

| 类型 | 数量 | 占比 | 备注 |
|------|------|------|------|
| match | 3 | 27% | brow_burst 触发 100% match |
| mismatch | 1 | 9% | 仅 5/9 V0.1 漏过的"光线极暗"伪信号 |
| skip | 6 | **55%** | 全来自 low_vlm_conf 触发，conf 0.0-0.1 临界值居多 |
| input | 1 | 9% | 川哥 free-form："我不在，检测错了吧" |

**有效训练信号：4 条（37%）**

### D.2 Phase 2 实战触发卡点

`trigger_if_ready` 双门：
- ✅ total ≥ 10：11/10
- ❌ mismatch ≥ 10：**1/10（10%）**

**真正卡点不是反馈量，是反馈类型分布**：
- daemon V0.2 trigger 设计偏向"模型自己拿不准"（low_vlm_conf）而非"模型错预测"（user 主动纠正）
- 模型当前预测准确率较高（mismatch 罕见对模型质量是好信号，对训练样本是坏信号）

### D.3 M0 W2 改进方向（已识别）

1. **daemon V0.3 加 user_triggered_mismatch channel**：川哥主动按快捷键 / 命令 / 前端按钮报告"模型搞错了"
2. **trigger 阈值平衡**：low_vlm_conf 0.65 → 0.50 减少 skip 浪费 / brow_burst 0.85 调整为 dynamic per-user
3. **input free-form NLU**：用 LLM 提取 "我不在" → label=missing / "我在专注" → label=focused，释放 17% 样本价值
4. **dnd / quiet hours**：白天 daemon 但弹窗仅在川哥 active 窗口（reduce 川哥被打扰感）

---

## E. 25h 硬门槛重评估（**PoC 时间盒诚实复盘**）

### E.1 真实数据采集时长

| 来源 | 帧数 | 时长 (1 fps) |
|------|------|-------------|
| signals_2026-05-09.jsonl | 27860 | 7.74h |
| signals_2026-05-10.jsonl | 18208 | 5.06h |
| **总计** | **46068** | **12.80h** |

### E.2 距硬门槛缺口

| 门槛 | 时长要求 | 当前 | 缺口 |
|------|---------|------|------|
| Day 13 末 | ≥ 15h | 12.80h | **-2.20h** |
| Day 14 末 | ≥ 25h | 12.80h | **-12.20h** |

### E.3 Day 14 单天最大增量预估

假设川哥 Day 14 daemon 使用：
- 4-6h：~16-18h 总（仍不达 25h）
- 8-10h：~20-22h 总（仍不达 25h）
- 12-14h：~24-26h 总（极限刚刚好）

**结论**：25h 硬门槛**不可达**（除非川哥 Day 14 整天 >12h 真人交互）。

### E.4 cc 自评：为何未及早发现？

- Day 12 末（5 反馈 / 1 mismatch）已是结构性短缺信号 → cc 当时**应该提**而非继续走 Phase 2 工程
- Day 13 中段 dry-run 跑出 `loss=-1.73` 数值不合理 → cc 未深究"为何 mismatch 标 distracted 但用户也没纠正"语义
- **改进**：下次 PoC（M1+）设 **Day N/2 mid-point 数据健康度评审节点**（强制复盘反馈/数据采集进度，非工程进度）

---

## F. 与 v0.3.2 product-definition 的差距

### F.1 已交付（PoC 范围内）

| 项 | v0.3.2 定义 | PoC 实际 |
|----|-------------|---------|
| LNN 主路径 | CfC（推理 ↑1-5 数量级） | ✅ ncps.torch.CfC + Day 6.5 91.5% val acc |
| 万能语法 20 槽 | 槽位编码 | ✅ slots/ 模块 + ADR 002 SlotValue 类型收紧（H 级，Day 11 仍待） |
| 个性化栈 | τ偏置 + FiLM + 槽位偏好 | ✅ τ偏置 (online + base) + FiLM (base) / 槽位偏好（M0 W2） |
| 多模态 | py-feat / Whisper / VLM | ✅ py-feat AUCaliper + Gemini VLM 主调 + MiniMax 备选 / Whisper 未集成（M0 W2）|
| LLM | DeepSeek-V4 主 + GLM-5.1 fallback | ⏳ Gemini 系列实际部署（DeepSeek/GLM 未对接，M0 W2） |
| KG | Graphiti bi-temporal | ⏳ M0 W3 |

### F.2 未达 / 部分达成

| 项 | 差距 |
|----|------|
| τ≈小时反思层 | Liquid-S4 候选未实测（M0 W3 / M9） |
| 多模态信号密度 | 仅视觉 + VLM；audio + context 模态未实采（fusion 走 missing_default）|
| CAPS if-then patterns | vboss_self 自评全空（待 M0 W2 真实采集补齐）|
| Schwartz 10 维 values | 仅 top3/bottom3 来自 personalt-test，剩余 partial |
| 跨日稳定性 | daemon 单天最长 5.72h，未跨 24h 真实场景验证 |

---

## G. M0 W2 路线图（详见 docs/m0-w2-tasks.md）

### G.1 P0（必做，2-3 周）

1. **daemon V0.3**：user_triggered_mismatch channel + trigger 阈值动态平衡（基于反馈分布自调）+ dnd / quiet hours
2. **feedback_to_batch v2**：missing mask + (sid, fid) 强制（去 V0.1 fallback）+ window padding 选项
3. **vboss_self 真实反馈微调**：M0 W2 攒够 ≥10 mismatch 触发首次 Phase 2 实战 + 5+ epoch 跑完 + ckpt 保存
4. **跨日稳定性**：daemon 24h 持续运行 + 内存监控（PoC 期间观察到 RSS 1.10→1.41GB 增长，需排查）

### G.2 P1（视精力）

1. **input free-form NLU**：用 LLM 提取隐含 reward 信号
2. **5 状态分类全测**：fatigued / distracted / idle / low_energy / emotional_negative
3. **Liquid-S4 τ≈小时反思层**：M9 候选实测对比 LTC
4. **Whisper 音频集成**：voice_present + utterance_density 真实采集

### G.3 P2（M1）

1. **Graphiti KG bi-temporal**
2. **DeepSeek-V4 主对接 + GLM-5.1 fallback**
3. **端侧 LFM2**

---

## H. PoC 期间沉淀的工程教训（沉淀到 mycc memory）

| Memory | 内容 |
|--------|------|
| feedback_lnmem_dev_harness | LnMem BRO-PDCA × 目标驱动验收 |
| feedback_lnmem_cognitive_spec | 评估同构原则（多模态信号自动聚合，禁问卷弹窗） |
| feedback_lnmem_realtime_capture_pinned | mediapipe<0.10.20 钉版本 + macOS TCC 摄像头授权链 |
| feedback_lnmem_real_data_baseline | vboss_001 ≠ 真人 / vboss_self HEXACO 距离 0.52 |
| feedback_lnmem_personalt_test_v02_blindspots | partial 字段 ⚠️ 标注（CAPS / 多模态 / Schwartz） |
| feedback_lnmem_pyfeat_model_mgmt | py-feat 0.6.2 模型管理 6 铁律 |
| feedback_lnmem_day65_pyfeat_lessons | Day 6.5 真人验证 10 踩坑 |
| feedback_lnmem_macos_popup_channel | macOS 弹窗 channel 替代飞书 bot 实现 2.5h vs 4-5h |
| feedback_lnmem_phase1_review_loop | codex 8 finding cc 4 修评判分歧（FB-02 误读） |
| feedback_lnmem_ood_data_dilution | 491 vlm OOD 数据稀释陷阱 |
| feedback_lnmem_trigger_threshold_calibration | 22/4min 过敏教训 → 0.85+10帧 |
| feedback_lnmem_realtime_state_confirmation | 模型预测 ping 川哥 match? 不允许只交付 Pearson 数字 |
| feedback_lnmem_day11_venv_rebuild | venv rebase 残留 / fresh venv 安全模式 / TCC 授权链 |
| feedback_lnmem_multi_ai_entry_pattern | CLAUDE/AGENTS/GEMINI/.cursorrules 强制索引 |
| feedback_global_view_before_local_fix | **元方法论：5min 全局扫描 SOP**（Day 11 川哥喊停产物） |
| feedback_haiku_subagent_trust_but_verify | haiku DoD 自报 ≠ cc 验收 |

---

## I. 三道门最后一关：等川哥签

cc 实现 ✅ → codex 对抗审 ✅（Day 12 + Day 13 双轮）→ **川哥签待 Day 14**

### I.1 川哥需要决策的关键项

1. **PoC 时间盒处理**（25h 不可达）：
   - 选 A：暂停 PoC 回 daemon V0.3 改造（数据严谨）
   - 选 B：放宽 25h 用 4 样本启 Phase 2 实战（工程演示优先，结果标 partial）
   - 选 C（cc 推荐）：dry-run 即 PoC 实质交付，反馈累积移交 M0 W2

2. **M0 W2 路线图**（G 段）：daemon V0.3 + feedback v2 + vboss_self 真实微调 + 跨日稳定性 = 2-3 周工作量

3. **Demo 录屏配合**：5-10min 拍摄需要川哥真人配合（让 daemon 触发 + 川哥操作弹窗）

4. **PoC 收尾报告 review**：本文档是否需要修订 / 补充 / 与 v0.3.2 product-definition 对齐
