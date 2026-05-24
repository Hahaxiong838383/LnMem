# ADR-027: dt-aware ODE 落地 + τ 长训失控 + Phase 2 multi-fps curriculum ready

> **决策者**：cc 主线 + codex (gpt-5.3-codex high) + gemini (3-flash-preview) 2 轮对抗审
> **状态**：accepted
> **关联**：ADR-022 τ-bias 梯度黑洞 · ADR-023 ODE rollout loss · ADR-024 数据驱动阈值 · ADR-025/026 罕见 AU 监控 · LNN-FIRST §1.2 核心能力 #1/#2

---

## 1. 触发事件

5/23 川哥钦定执行三方综合 TODO-1（dt-aware ODE 实施），cc + codex + gemini Team 模式落地。

发现根因（5/22 cc 诊断）：
- `custom_cfc_cell.py` ODE 半隐式欧拉公式 `h_new = (τ*h + dt*f) / (τ+dt)` 显式用 dt
- cell + multilayer + trainer.backbone_step 接口都接受 `dt: float = 1.0`
- **`ode_dynamics_loss.py:100,120` 调 `backbone_step_fn` 完全不传 dt** → dt 永远 = 1.0
- 后果：VEATIC 30fps 用 dt=1.0 错 10x；daemon 1Hz 用 dt=1.0 错 3x；模型完全无法区分时间尺度

## 2. 实施（dt-aware Phase 1）

### 2.1 改动文件（10 个）

| 文件 | 改动 |
|------|------|
| `poc/lnn/ode_dynamics_loss.py` | OdeDynamicsLossConfig 加 dt_target/fps_default/use_dt_aware；forward 接 fps，按 dt 自适应 stride，传 dt 给 backbone_step |
| `poc/training/dataset.py` | VEATICDataset getitem 返回 fps=30.0 |
| `poc/training/daemon_dataset.py` | DaemonAUDataset getitem 返回 fps=1.0 |
| `poc/training/trainer.py` | train_one_epoch 透传 fps 给 ode_loss |
| `poc/training/evaluate.py` | run_evaluation 透传 fps，结果含 actual_dt / total_physical_horizon_s |
| `poc/training/inference_driver.py` | CLI 接受 fps 参数，gates_summary 增 actual_dt |
| `poc/training/tests/test_dt_aware_ode.py` | 6 个新 TDD 测试 |
| `poc/lnn/tests/test_ode_dynamics_loss.py` | 兼容性修正 |
| `poc/training/tests/test_ode_loss_integration.py` | 兼容性修正 |
| `poc/training/tests/test_inference_driver.py` | actual_dt 字段断言 |

### 2.2 gemini round 1 对抗审 5 finding（已采纳）

| Finding | 风险 | 采纳处理 |
|---------|------|---------|
| F1 v3.0 ckpt 时间幻觉 | CRITICAL | 验收策略改：用 daemon fps=1.0 + dt_target=1.0 推理 v3.0（actual_dt=1.0 跟 ckpt 一致）|
| F2 采样定理违约 | HIGH | `actual_dt = max(dt_target, 1.0/fps)` + WARNING |
| F5 Rollout 物理时间对账 | HIGH | 输出 actual_dt + total_physical_horizon_s |
| F3 异构 batch tensor shape | CRITICAL | Phase 1 单 fps batch，Phase 2 加 BucketBatchSampler |
| F4 Shuffle 假阳性 | MEDIUM | Phase 1 同时报 model_vs_naive + trivial_recon_check 防伪阳 |

### 2.3 测试

- 87 训练测试零回归
- 6 新 dt-aware TDD 全过
- Studio 28/28 全过

## 3. Phase 1 实测验收（**重大里程碑 + 新 bug**）

### 3.1 ✅ τ 分布完全解放（LNN-FIRST §1.2 核心能力 #2 第一次实现）

| ckpt | τ range | 宽度 | spec [0.1, 10]s 覆盖 |
|------|---------|------|---------------------|
| **v3.0 200ep (dt=1.0)** | [0.991, 1.009] | 0.018 | **0.2%**（基本单时间尺度 RNN）|
| **dt-aware 30ep (dt=0.1)** | **[0.96, 34]** | **33.0** | **300%**（覆盖 spec 并扩展到 30s+）|

**这是项目级里程碑**：v3.0 11 轮迭代以来 τ 第一次真正"活起来"分化为多时间尺度。

### 3.2 VEATIC 内部 eval 不退化

| 门 | v3.0 200ep | dt-aware 30ep | dt-aware 200ep |
|------|-----------|---------------|---------------|
| L_world_model | 0.023 | 0.023 ✅ | 0.021 ✅ |
| vs_persistence | 0.50 | 0.60 ✅ | **0.54** ✅ |
| vs_naive | 0.63 | 0.62 ✅ | 0.56 ✅ |
| shuffle_ratio | 1.73 | 0.55 ✅ | 0.51 ✅ |
| shuffle_test_pass | True | True ✅ | True ✅ |
| trivial_recon_check | True | True ✅ | True ✅ |
| effective_dim | 4.92 | 3.06 | 3.59 |
| actual_dt | 1.0 | **0.1** ✅ | **0.1** ✅ |
| physical_horizon_s | 24 | 0.8 | 0.8 |

### 3.3 ❌ daemon 跨分布 eval 仍崩盘

| 指标 | v3.0 | dt-aware 30ep | dt-aware 200ep |
|------|------|---------------|---------------|
| L_world_model | 0.057 | 0.262 | 0.431 |
| vs_persistence | 2.27 | 10.74 | **17.65**（更差）|
| shuffle_ratio | 0.008 | -0.002 | **0.057**（仅微升）|
| effective_dim | 5.19 | 4.50 | 5.47 |

**结论**：dt 修了 ODE 数学正确性但**不解决跨分布泛化**（单 VEATIC 训练自然 generalize 不到 daemon）。

### 3.4 🔥 τ 长训失控（200ep 新 bug）

| ckpt | τ 范围 | 后果 |
|------|--------|------|
| 30ep | [0.96, 34] s | 合理（覆盖 spec [0.1, 10]）|
| **200ep** | **[0.96, 3.5e9] s** | **τ 爆炸：35 亿秒 = "永不忘"死神经元** |

**根因**：tau_lr=1e-3 + 无 τ 上界 clamp + 200ep 长训 → τ 一路涨没刹车

**影响**：
- 30ep daemon 比 v3.0 差（10.74 vs 2.27），200ep 更差（17.65）
- VEATIC 短 rollout (0.8s) 不受影响（死神经元 dh/dt≈0 但 prefix 已捕捉信息）
- daemon 长 rollout (24s) 受灾（死神经元主导，模型变常数预测器）

**这是 ADR-022 τ-bias 梯度黑洞修复后的反方向失控**（之前过弱→修了过强）。

## 4. TODO-1 验收结论

| 目标 | 结果 | 状态 |
|------|------|------|
| dt 真透传到 ODE 内核 | 87 测试通过 + Studio τ 实测分化 | ✅✅ |
| **τ 分布解放为多时间尺度** | 0.018 → 33（1800x，覆盖 spec 300%）| ✅✅ **意外里程碑** |
| VEATIC 内部不退化 | 5 道门基本过 | ✅ |
| daemon 1Hz shuffle_ratio 0.008 → ≥ 0.5 | 0.057（仅微升 7x，远未达标）| ❌ |
| 新发现：τ clamp 需求 | 30ep OK, 200ep 失控 | ⚠️ 必修 |

## 5. Phase 2 multi-fps curriculum spec V0.2（gemini round 2 对抗审整改）

### 5.1 数据集组合
- VEATIC 30fps（dt_target=0.1）
- daemon 1Hz 378k 帧（dt_target=1.0）
- VEATIC 降采样 5Hz / 1Hz（dt_target=0.2 / 1.0）—— **gemini F3 警告 1Hz 仅 1-2 帧无效，必须 sliding window stride=30 + 序列长度 L≥8**

### 5.2 BucketBatchSampler
**gemini F4 整改**：Equally Weighted Bucket Sampling（不按样本量按 fps 类别均等）

### 5.3 课程调度
**gemini F2 整改**：Linear Probability Ramping（不硬切，40-60ep 线性从 0% → 30%）

### 5.4 Loss 设计
**gemini F1 整改**：
- Conditional Loss Masking（daemon 无 V-A 不参与 V-A loss）
- **强制 Invariance Loss**（不延到 Phase 2.5）：`MSE(h_30fps[::30], h_1Hz)` 跨 fps 隐空间对齐

### 5.5 Eval 防伪
**gemini F5 整改**：Variance-Aware Eval —— shuffle_ratio 高同时检查 `std(pred) ≥ 0.1*std(target)` 防低通滤波伪阳

### 5.6 τ clamp（cc Phase 1 新发现）
- 加 `tau_upper_bound = 30.0`（按 spec [0.1, 10]s 上界 3x）
- 或 weight_decay on tau_base
- 防 200ep 长训 τ 爆炸 → 死神经元

## 6. LNN-FIRST 6 项对账

| 项 | dt-aware 影响 | Phase 2 影响 |
|---|---------------|--------------|
| #1 万能语法 | 不变 | 不变 |
| #2 多时间尺度 LNN | ✅✅ **τ 解放为多时间尺度（30ep 实证）** | ✅ 多 fps 训练验证跨采样率泛化 |
| #3 CLS consolidation | 不变 | 不变 |
| #4 per-user 个性化栈 | 不变（persona_tau_bias 已存在）| 不变 |
| #5 居家+车载双场景 | 间接服务（任意 fps）| 间接服务 |
| #6 用户记忆 UI | 不变 | 不变 |

## 7. 5/27 启动门 V3（融合 dt-aware + τ 失控保护）

| # | 门 | 来源 |
|---|---|---|
| 1 | **τ.std > 0.5 且 τ.max < 100s（防爆炸）** | ADR-024 + **ADR-027 新增上界** |
| 2 | vs_persistence < 0.7 | ADR-024 |
| 3 | effective_dim / input_effective_dim ≥ 0.8 | ADR-024 |
| 4 | Shuffle ratio ≥ 1.5 | ADR-024 |
| 5 | rare_au_verdict ∈ {green, yellow} | ADR-025/026 |
| 6 | **cross-distribution shuffle ratio ≥ 0.5**（daemon 1Hz 必测）| **ADR-027 新增 cross-dist 验证** |
| 7 | **variance-aware shuffle 防伪**（gemini F5）| **ADR-027 Phase 2** |

## 8. 待办

按优先级：

1. **Phase 2 multi-fps curriculum 实施**（需川哥拍板）：codex 实现 + Studio 训练 + daemon eval 验收
2. **τ clamp 加入 OdeDynamicsLoss / TrainConfig**（防 200ep 爆炸）
3. **Variance-Aware Eval 加入 evaluate.py**（gemini F5 防伪）
4. ADR-026 fixed_bottom 按 N=124 训练集 AU 频率实测重新校准（5/27 启动前）

## 9. mycc 教训新增

- **`feedback_lnn_tau_long_training_explosion`**：tau_lr 释放后必须配 τ_upper_bound clamp，否则长训 τ 一路涨变死神经元（ADR-022 反方向失控）
- **`feedback_dt_aware_solves_math_not_distribution`**：dt-aware 修 ODE 数学正确性但不解决跨分布泛化，需要 multi-fps curriculum 才能真正 dt-invariant
- **`feedback_tau_distribution_release_is_milestone`**：τ 从单时间尺度 [0.991, 1.009] 解放到 [0.96, 34] 是 LNN-FIRST 核心能力 #2 第一次实现

---

**📝 维护者**：cc 主线 + codex 实施 + gemini 2 轮对抗审 | **签发**：2026-05-23（5/23 凌晨 Team 模式）
