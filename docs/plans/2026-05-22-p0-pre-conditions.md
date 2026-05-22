# P0 5 项 Pre-conditions 实施契约（2026-05-22 v2 整改后）

> Team 模式：cc（规划+PC-3+终审）/ codex（PC-1/2/4/5 代码）/ gemini-3.5（对抗评审 ×2）
> 目标：5/27 Phase γ Step 1 启动前必须 5 项 PC 全绿 + effective_dim ≥ 8.0
> 上游：`docs/retrospectives/2026-05-21-day3-evening-team-mode-retro.md` §5 + `ROADMAP.md` §5/27
>
> **v2 整改 = gemini 第 7 轮前置评审 5 findings 全采纳**：
> - F-1 CRITICAL：cov 孤立 → 自我静音作弊 → PC-3 改 std+cov 联合
> - F-2 HIGH：硬 latch → 软门控 sigmoid(4·(2.5-d))
> - F-3 HIGH：tau_lr=1e-5+DGB 双重锁死 → 改 tau_lr=1e-3 让 DGB 单项控制
> - F-4 HIGH：缺 N=124 工程防御 → 新增 PC-5
> - F-5 MEDIUM：τ.grad 必须在 backward 后 / zero_grad 前捕获

---

## 共同前提（所有协作方必读）

- 仓库：`/Users/rayman.chen/mycc/2-Projects/LnMem-poc`
- 分支：`poc/lnn-validation`
- 当前 HEAD：`6b0fe73`
- 核心文件：
  - `poc/training/trainer.py` (TinyTrainer + TrainConfig + ADR-022 DGB)
  - `poc/training/evaluate.py` (effective_dim SVD + shuffle test)
  - `poc/lnn/losses.py` (InfoNCELoss + VICRegLoss + VICRegConfig)
  - `poc/training/run_tiny_training.py` (CLI entrypoint)
- 不可妥协 6 项：LNN-FIRST / per-user τ-bias / AU 多模态 / 离线优先 / ODE 可解释 / 时序记忆
- 中文回复 + 任务署名
- **禁止破坏现有 22 finding 修复**（cherry-pick 安全垫 / freeze 梯度泄露 / ADR-022 DGB）

---

## PC-1 τ-Monitor（codex 后台）

### 契约
每 epoch 结束时记录 τ 统计，写入 `epoch_logs` 列表，CLI 输出含 τ 字段。

### 实施位置
- `poc/training/trainer.py::TinyTrainer.train_one_epoch()` 末尾
- `poc/training/run_tiny_training.py` CLI 输出新增 τ 行

### 数据格式
每 epoch log 末尾新增字段：
```python
{
    "epoch": int,
    "loss_mean": float,
    # ... 现有字段 ...
    # 新增：
    "tau_stats": {
        "mean": float,         # τ_i 跨 neurons 平均（应 ∈ [0.05, 0.2]）
        "min": float,
        "max": float,
        "std": float,
        "grad_norm": float,    # τ-bias 累积梯度 norm（应 > 0，证明 LR=1e-5 在学）
    }
}
```

### 验收（v2 修正：cc PLAN v1 把 spec 范围写错了，实际 spec base-v3 §86）
- 每 epoch print: `τ mean=1.000 ± 0.004 ∈ [0.994, 1.006] grad=1.8e-2`（实测）
- τ.mean ∈ [0.1, 10] s（spec base-v3 §86 钦定 Layer 1 范围）
- τ.grad_norm > 1e-5（PC-4 v3 释放 tau_lr 后，DGB 后仍 ≥ 1e-5 量级证明可学）
- 单测：`poc/training/tests/test_tau_monitor.py`，至少 2 条用例（mock trainer / 真 trainer 1 step）
- **注**：cc PLAN v1 误写 [0.05, 0.2] 是查错 spec 段落（实际 §9 没这个数字，§86 才有 Layer 1/2 时间常数）

### F-5 整改：grad 捕获时序（gemini 第 7 轮 MEDIUM）
**严格时序**：必须在 `loss.backward()` 之后、`optimizer.step()` / `optimizer.zero_grad()` **之前**捕获 τ.grad：
```python
# trainer.py train_one_epoch 内 step 循环：
loss.backward()
# ✅ 必须在这里捕获 τ.grad（detach + clone + 累积到 buffer）
for n, p in self.model.named_parameters():
    if "persona_tau_bias" in n and p.grad is not None:
        self._tau_grad_buffer.append(p.grad.detach().norm().item())
# ADR-022 DGB
self._apply_dgb()
self.optimizer.step()
self.optimizer.zero_grad()  # 之后 .grad 全清，再读就是 0/None
```
**反模式**（禁止）：在 `optimizer.step()` 或 `zero_grad()` 之后读取 `.grad`

### 不准踩
- 不准改 ADR-022 双率优化器结构
- 不准改 DGB 逻辑
- τ.grad 计算时确保 backward 已经发生（每个 step 取，epoch 末 aggregate）

---

## PC-2 DataLoader shuffle=True（codex 后台）

### 现状（必读）
`trainer.py::train_one_epoch` 没用 DataLoader，是手写：
```python
all_windows = []  # 一次性把所有视频切窗
for i in range(steps_per_epoch):
    indices = torch.randperm(len(all_windows))[:batch_size * 4]
    sampled_windows = [all_windows[i] for i in indices]
```
**问题**：每 epoch 内 windows pool 顺序固定，randperm 用全局 seed 时**每个 epoch 的 batch 序列完全相同**。

### 契约
1. 引入 `torch.utils.data.DataLoader` 包装 windows pool
2. `shuffle=True` 显式声明
3. 每 epoch reseed（用 `epoch_idx` 作为 generator seed offset）
4. 加 sanity check：`epoch 0` vs `epoch 1` 第一个 batch 的 `video_id` 序列必不同
5. Epoch 1 vs Epoch 2 loss diff ≤ 5%（PC retro 钦定，证明数据顺序变了但 loss 收敛仍稳）

### 实施
- 新建 `poc/training/window_dataset.py`：
  ```python
  class WindowDataset(torch.utils.data.Dataset):
      def __init__(self, all_windows): ...
      def __getitem__(self, idx): return all_windows[idx]
      def __len__(self): return len(all_windows)
  ```
- 修改 `TinyTrainer.train_one_epoch`：
  ```python
  windows_ds = WindowDataset(all_windows)
  loader = DataLoader(
      windows_ds,
      batch_size=self.cfg.batch_size * 4,  # 保留 contrastive 多抽样
      shuffle=True,
      generator=torch.Generator().manual_seed(self.cfg.seed + epoch_idx),
      collate_fn=lambda x: x,  # 不做 stacking，由 _build_batch 处理
  )
  for sampled_windows in loader:
      batch = self._build_batch(sampled_windows)
      # ...
  ```

### 验收
- 加诊断 print：`epoch 0 first batch vids: ['v01','v07','v03',...]` / `epoch 1 first batch vids: ['v05','v02','v09',...]`，至少 50% 元素不同
- 单测 `test_dataloader_shuffle.py`：连续 3 个 epoch 的第一个 batch 索引序列两两不同
- 跑 1 epoch dry-run，loss 仍 < baseline 1.5x（不破坏既有训练）

### 不准踩
- 不改 batch 构造逻辑（`_build_batch` 内部不动）
- 不改 contrastive pair 抽取算法（同 vid anchor/pos + cross-vid neg）
- 不引入新依赖

---

## PC-3 动态 VICReg Std+Cov auto-trigger（cc 主力，v2 整改完成）

### 契约 v2（gemini F-1/F-2 整改后）
当 `effective_dim` 偏低时，自动在 InfoNCE 主损失之上**叠加** VICReg 的 **std + cov 联合项**（不是 cov 孤立用，那会触发自我静音作弊），权重用**软门控**而非硬 latch：

### 数学
```
L_total = L_InfoNCE + λ_cov(effective_dim) · (std_weight · L_std + cov_weight · L_cov)

λ_cov(d) = sigmoid(gate_steepness · (threshold - d))
  - d=2.5 (threshold)：λ = 0.5
  - d=1.0：λ ≈ 0.998
  - d=4.0：λ ≈ 0.0025
  - d=8.0：λ ≈ 0（effectively off）

std_weight = cov_weight = 1.0（gemini F-1 钦定 1:1）
gate_steepness = 4.0（gemini F-2 钦定）
```

### 实施位置（cc 已完成）
- 新文件 `poc/training/vicreg_trigger.py`（VICRegCovTrigger + VICRegCovTriggerConfig）
- 单测 `poc/training/tests/test_vicreg_auto_trigger.py` 19/19 通过
- **trainer.py 集成点**（codex 重派后协助 wiring）：
  ```python
  # TinyTrainer.__init__
  self.vicreg_trigger = VICRegCovTrigger(VICRegCovTriggerConfig())

  # train step 内（在 loss.backward 之前）
  loss = self.loss_fn(anchor_h, positive_h, negatives_h)
  lam = self.vicreg_trigger.current_lambda
  if lam > 1e-3:
      aux = self.vicreg_trigger.compute_loss(anchor_h, positive_h)
      loss = loss + lam * aux["loss"]

  # train_one_epoch 末尾跑完 evaluate
  newly = self.vicreg_trigger.update(effective_dim=eval_result["effective_dim"], epoch=epoch_idx)
  if newly:
      print(f"⚠️  VICReg Std+Cov AUTO-TRIGGERED (effective_dim={eval_result['effective_dim']:.2f})")
  epoch_log.update(self.vicreg_trigger.to_log())
  ```

### 验收（cc 已 19/19）
- ✅ 软门控 λ 在 threshold=2.5 处 = 0.5
- ✅ λ 随 effective_dim 单调递减
- ✅ disabled 时 λ ≡ 0
- ✅ std loss 惩罚维度静音（F-1 整改核心验证）
- ✅ 自我静音作弊比正常更贵（loss_silenced > loss_normal）
- ✅ 与 VICRegLoss 内部 std/cov 计算公式严格一致

### 不准踩
- 不改 InfoNCELoss 内部
- 阈值 2.5 / patience 5 严格按 retro 钦定
- 不能改成 cov 孤立（会触发 F-1 自我静音作弊）

---

## PC-4 τ LR=1e-3 + DGB 单项控制（v2 整改，gemini F-3 HIGH）

### 现状 + 改判轨迹
- gemini 第 5 轮：`tau_lr=0.0`（完全冻结，安全但学不到）
- gemini 第 6 轮：`tau_lr=1e-5 + DGB`（终审，担心 underflow → 保留学习能力）
- **gemini 第 7 轮（本轮）F-3**：**1e-5 + DGB = 双重锁死 = 伪学习**
  - 数学：DGB 已经把 τ.grad 缩到 main_grad/10，再叠 1e-5 LR → 实际更新 ~1e-6~1e-7 级
  - τ 初值 0.1，整训练期累积位移 < 1%，对时序自适应**无意义**
  - 这是为了通过 PC-1 `grad_norm > 1e-9` 验收的「表面合规」

### 契约 v2
- **`TrainConfig.tau_lr` 从 0.0 → 1e-3**（不是 1e-5，让 DGB 单项控制）
- `_apply_dgb` 保留并主导（safety net + 主控）
- DGB target_ratio 仍 10.0（bias_grad ≤ main_grad / 10）
- 注释更新：`# ADR-022 v3 (gemini 第 7 轮终审): 释放 tau_lr，让 DGB 单项控制`

### 验收
- 跑 1 epoch dry-run：
  - τ.grad_norm > 0（非死参数）
  - **τ.grad_norm 较大**（≥ 1e-5 量级，证明不是 underflow）
  - bias_grad/f_net_grad post-DGB ≤ 0.1（健康范围）
  - **τ.item 在 epoch 内有可测位移**（mean diff > 1e-5）
- 单测：`test_tau_lr_1e_3_dgb.py` 验证上述 4 条 + DGB scale 仍然生效

### 不准踩
- 不准回到 0.0 或 1e-5（gemini 第 7 轮已改判）
- 不准移除 DGB（它是主控）
- ADR-022 文档同步更新：v2 → v3

---

## PC-5 工程防御 in-memory cache + Truncated BPTT（v2 新增，gemini F-4 HIGH）

### 背景
gemini 第 7 轮 F-4 抓出 PLAN 只看「指标合规」忽视「N=10 → N=124 的工程灾难」：
1. **I/O 阻塞**：每 epoch 从磁盘读 124 视频的 AU 特征，磁盘 I/O 会让训练时间从 4 天 → 8 天+
2. **ODE 显存爆炸**：LNN BPTT 沿时间步反传，长序列在 N=124 易触发 OOM

### 契约
#### PC-5a：WindowDataset 内存预缓存
- `WindowDataset.__init__` 一次性将所有窗口 load 进 RAM
  ```python
  class WindowDataset(Dataset):
      def __init__(self, all_windows):
          # 一次性把 event_tensor 全部 stack 到 CPU 内存（M2 Max 内存充足）
          self._cache = [
              {
                  "event_tensor": w["event_tensor"].cpu().pin_memory()
                                  if torch.cuda.is_available()
                                  else w["event_tensor"].cpu(),
                  "video_id": w["video_id"],
                  "persona_idx": w.get("persona_idx", 0),
              }
              for w in all_windows
          ]
  ```
- 防御：N=124 估算（30 帧/窗，10 窗/视频，52 维 float32 → 6KB/窗 → 124*10*6KB ≈ 7.5MB，远小于 96GB M2 Max RAM）

#### PC-5b：Truncated BPTT 上限
- `TrainConfig` 新增 `max_bptt_steps: int = 32`
- `MultiLayerCfC.forward` 接受 `bptt_truncate` 参数，每 `max_bptt_steps` 步 `h.detach()` 切断反向传播
  ```python
  for t in range(seq_len):
      h = self.step(x[:, t], h)
      if t > 0 and t % bptt_truncate == 0:
          h = h.detach().requires_grad_(True)  # 截断 BPTT
  ```
- 默认 32 步（约 1 秒视频片段），在 InfoNCE 任务下足够

### 验收
- 单测 `test_window_dataset_cache.py`：缓存后第 2 次 access 不读磁盘
- 单测 `test_truncated_bptt.py`：每 32 步 h 在 grad graph 上是新 leaf
- 1 epoch dry-run：峰值 RSS < 2GB（N=10 时），训练时间不退化
- 注释：在 `TrainConfig` 标注 `# gemini 第 7 轮 F-4 工程防御`

### 不准踩
- 不准把 cache 放 GPU（会 OOM）
- 不准移除 BPTT 上限（5/27 N=124 会爆）
- 不准在 N=10 跑时让 cache 变慢（容量检查 < 1% RAM 即直接缓存）

---

## 终审清单（cc 跑）

完成 PC-1..5 后必须全过：

1. ✅ `pytest poc/` 全绿（既有 679 + 新增 ~30 测试）
2. ✅ 1 epoch dry-run loss 不 > 既有 baseline 1.5x
3. ✅ τ stats print 出现且 ∈ [0.1, 10] s（spec base-v3 §86，默认 init τ ≈ 1.0）
4. ✅ DataLoader shuffle 诊断 print 显示 vid 序列 epoch 间不同
5. ✅ τ LR=**1e-3** 时 grad_norm ≥ 1e-5（不是 underflow 级 1e-9）
6. ✅ VICReg auto-trigger 软门控 λ 在 effective_dim=2.5 时 = 0.5
7. ✅ WindowDataset 缓存验证（PC-5a）
8. ✅ Truncated BPTT 切断 grad graph（PC-5b）

---

## Team 协作时间线（目标 5/22 内完工）

| T+0 | T+10min | T+40min | T+60min | T+90min |
|---|---|---|---|---|
| cc 写完 PLAN | codex 后台启动 PC-1/2/4 + cc 写 PC-3 | codex 完成 | cc 整合 + 跑 dry-run | gemini 第 7 轮终审 + commit |

## gemini 第 7 轮终审输入

完成所有 PC 后，给 gemini-3.5-flash + reasoning_effort=high + max_tokens=65535 的输入：

```
你是 LnMem 项目第 7 轮终审 reviewer。
前 6 轮你已经审过（CONDITIONAL APPROVE），见 docs/retrospectives/2026-05-21-day3-evening-team-mode-retro.md §2 §3。

本轮审查 5/22 落地的 4 项 Pre-conditions（PC-1..4）：
1. τ-Monitor（PC-1）
2. DataLoader shuffle（PC-2）
3. VICReg Cov auto-trigger（PC-3）
4. τ LR=1e-5+DGB（PC-4）

附件：完整 git diff `6b0fe73..HEAD`、4 个 PC 单测、1 epoch dry-run 日志。

请按 6 轮风格输出：
- finding 列表（CRITICAL/HIGH/MEDIUM/LOW）
- 每个 finding 含「现象/根因/修复建议/严重度」
- 最终 verdict：APPROVE / REVISE / BLOCK
- 5/27 Phase γ Step 1 是否可启动
```

---

## 风险/盲区登记

- **Studio 公网可达性**：50 epoch 重测需 Studio M2 Max，川哥在家走雷电桥 ProxyJump，需 Mini 在线
- **VICReg cov 单独使用未实测**：仅用 cov 项不带 std/sim 可能数学上退化，PC-3 单测必须覆盖这点
- **τ LR=1e-5 与 frozen 等效性**：retro §3 担心 1e-5 几乎等效冻结，PC-4 单测必验 grad_norm > 1e-9

---

**起始时刻**：2026-05-22 23:30（5/21 实际记账 + cc 校正时钟）
**目标完工**：2026-05-23 凌晨 01:30 内
**主审**：cc（claude）
**协审**：codex（gpt-5.3-codex high）/ gemini-3.5-flash（reasoning_effort=high）
