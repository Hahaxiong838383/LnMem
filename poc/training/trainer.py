"""γ-3 Step 3: tiny VEATIC 1 epoch Trainer（spec §5 训练计划 + gemini F-4.1 防 shortcut）

最小可跑训练循环，验证管道端到端 work（不追求最优超参）.

设计要点（gemini F-4.1 + spec v0.2.8）:
- AdamW + f_net weight_decay=1e-3 防 shortcut learning
- persona_tau_bias LR scaling 5x 主干 LR
- freeze f_net 前 1k step 强迫先学 τ-bias（spike 简化为 200 step）
- 每 100 step 监控 bias_grad/f_net_grad ratio < 1e-3 报警
- collapse 四联实时监控
"""
from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from poc.lnn import MultiLayerCustomCfC
from poc.lnn.heads import FusionLayer, FusionConfig
from poc.lnn.losses import InfoNCELoss, InfoNCEConfig, VICRegLoss, VICRegConfig, detect_semantic_breakpoints
from poc.lnn.ode_dynamics_loss import Decoder, OdeDynamicsLoss, OdeDynamicsLossConfig
from poc.monitoring.collapse_monitor import CollapseMonitor
from .dataset import VEATICDataset, split_to_windows
from .au_augment import au_augment, AUAugmentConfig
from .vicreg_trigger import VICRegCovTrigger, VICRegCovTriggerConfig
from .window_dataset import WindowDataset


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """spec §5 训练超参 + ADR-022 梯度黑洞修复.

    实测发现（5/22 04:00 γ-3 spike4）：bias_grad/f_net_grad = 6.4e8 反向异常
    →τ-bias 路径 O(1) 短路吞噬 f_net 长链梯度（gemini ADR-022 钦定）。
    """
    seq_len: int = 30
    batch_size: int = 16
    num_negs: int = 16
    hidden_dim: int = 64
    num_layers: int = 2
    num_personas: int = 4
    epochs: int = 1
    lr: float = 1e-3
    # ADR-022 v3 (gemini 第 7 轮终审): 释放 tau_lr，让 DGB 单项控制
    tau_lr: float = 1e-3
    dgb_target_ratio: float = 10.0
    dgb_enabled: bool = True
    persona_lr_scale: float = 1.0       # DEPRECATED by ADR-022 双率优化器
    f_net_weight_decay: float = 1e-3    # gemini F-4.1
    freeze_f_net_steps: int = 200       # spike 简化（spec 1k step）
    max_bptt_steps: int = 32            # gemini 第 7 轮 F-4 工程防御
    use_vicreg: bool = False            # False = InfoNCE 主路径
    use_ode_loss: bool = True           # ADR-023 主路径
    use_breakpoint_mask: bool = True    # gemini 终审条件 1
    log_every_n: int = 50
    seed: int = 42
    # 硬门 #3 (gemini 第 5 轮): AU augmentation 替代 spatial aug
    au_augment_enabled: bool = True
    au_dropout_p: float = 0.3        # 实测甜区：M5 833→500（-40%）, Shuffle drop +0.241
    au_jitter_std: float = 0.05      # 同上
    au_temporal_mixup_alpha: float = 0.0  # 关：激进 mixup 反而劣化（实测 5/21）


class TinyTrainer:
    """tiny VEATIC 1 epoch 训练（含 collapse 监控）"""

    def __init__(self, cfg: TrainConfig, dataset: VEATICDataset, device: str = "cpu"):
        self.cfg = cfg
        self.dataset = dataset
        self.device = torch.device(device)
        torch.manual_seed(cfg.seed)

        # 模型：FusionLayer → MultiLayerCustomCfC → 直出 hidden（spike 不接 readout heads）
        self.fusion = FusionLayer(FusionConfig(input_dim=52, fusion_dim=cfg.hidden_dim)).to(self.device)
        self.backbone = MultiLayerCustomCfC(
            input_dim=cfg.hidden_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_personas=cfg.num_personas,
        ).to(self.device)

        # 损失
        if cfg.use_vicreg:
            self.loss_fn = VICRegLoss(VICRegConfig()).to(self.device)
        else:
            self.loss_fn = InfoNCELoss(InfoNCEConfig(learnable_sigma=False)).to(self.device)
        self.decoder = Decoder(hidden_dim=cfg.hidden_dim, input_dim=52, num_layers=1).to(self.device)
        self.ode_loss = OdeDynamicsLoss(
            cfg=OdeDynamicsLossConfig(),
            decoder=self.decoder,
        ).to(self.device)

        # collapse 监控
        self.monitor = CollapseMonitor()
        self.vicreg_trigger = VICRegCovTrigger(VICRegCovTriggerConfig())

        # AdamW 分组：f_net 加 weight_decay + persona_tau_bias 独立 LR
        f_net_params = []
        persona_bias_params = []
        other_params = []
        for cell in self.backbone.cells:
            for n, p in cell.named_parameters():
                if "persona_tau_bias" in n:
                    persona_bias_params.append(p)
                elif "f_net" in n:
                    f_net_params.append(p)
                else:
                    other_params.append(p)
        for p in self.fusion.parameters():
            other_params.append(p)
        if cfg.use_ode_loss:
            for p in self.decoder.parameters():
                other_params.append(p)

        # ADR-022 v3: 双率优化器（τ-bias 释放到 1e-3，主干由 DGB 主控约束）
        self.persona_bias_params = persona_bias_params
        self.f_net_params = f_net_params
        self.optimizer = AdamW([
            {"params": other_params, "lr": cfg.lr, "weight_decay": 0.0},
            {"params": f_net_params, "lr": cfg.lr, "weight_decay": cfg.f_net_weight_decay},
            {"params": persona_bias_params, "lr": cfg.tau_lr, "weight_decay": 0.0},
        ])

        self.step = 0
        self.logs = []
        self.epoch_logs: list[dict[str, Any]] = []
        self._last_first_batch_vids: Optional[list[str]] = None

    def encode(self, x: torch.Tensor, persona_idx: torch.Tensor) -> torch.Tensor:
        """52d → 64d hidden seq."""
        x_fused = self.fusion(x)  # (B, T, 64)
        h_seq = self.backbone(
            x_fused,
            persona_idx,
            bptt_truncate=self.cfg.max_bptt_steps,
        )
        return h_seq

    def _backbone_step_for_ode(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        persona_idx: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """Single-step helper for ODE losses; x_t uses raw 52d input."""
        x_fused = self.fusion(x_t)
        if self.backbone.num_layers == 1:
            return self.backbone.cells[0](x_fused, h_prev, persona_idx, dt=dt)
        # 两层近似：layer-1 走短记忆状态（zero init），layer-2 用真实 h_prev
        h1_prev = torch.zeros_like(h_prev)
        h1 = self.backbone.cells[0](x_fused, h1_prev, persona_idx, dt=dt)
        return self.backbone.cells[1](h1, h_prev, persona_idx, dt=dt)

    def _freeze_f_net_if_needed(self):
        """gemini F-4.1: 前 N step freeze f_net 强迫先学 τ-bias."""
        freeze = self.step < self.cfg.freeze_f_net_steps
        for cell in self.backbone.cells:
            for p in cell.f_net.parameters():
                p.requires_grad = not freeze

    def _build_contrastive_pairs(self, windows: list[dict]) -> Optional[dict]:
        """从切片窗口构造 (anchor, positive, negatives) batch.

        spike 阶段简化策略：
        - anchor = 窗口最后一帧 hidden
        - positive = 同视频下一个窗口最后一帧 hidden
        - negatives = 不同视频窗口最后一帧 hidden
        """
        # 按 video_id 分组
        from collections import defaultdict
        by_vid = defaultdict(list)
        for w in windows:
            by_vid[w["video_id"]].append(w)

        anchor_list, pos_list = [], []
        vids_kept = []
        for vid, ws in by_vid.items():
            if len(ws) < 2:
                continue
            # 随机选 anchor 窗口，positive 是同视频另一窗口
            idx_a = torch.randint(0, len(ws) - 1, (1,)).item()
            anchor_list.append(ws[idx_a]["event_tensor"])
            pos_list.append(ws[idx_a + 1]["event_tensor"])
            vids_kept.append(vid)

        if len(anchor_list) < 2:
            return None  # 不够 batch

        anchor = torch.stack(anchor_list).to(self.device)  # (B, T, 52)
        positive = torch.stack(pos_list).to(self.device)

        # negatives: 每个 anchor 配 num_negs 个跨视频窗口
        all_windows = windows
        neg_list = []
        for i, vid in enumerate(vids_kept):
            cross_vid_windows = [w for w in all_windows if w["video_id"] != vid]
            if len(cross_vid_windows) < self.cfg.num_negs:
                neg_idx = torch.randint(0, len(cross_vid_windows), (self.cfg.num_negs,))
            else:
                neg_idx = torch.randperm(len(cross_vid_windows))[:self.cfg.num_negs]
            neg_tensors = torch.stack([cross_vid_windows[j]["event_tensor"] for j in neg_idx])
            neg_list.append(neg_tensors)
        negatives = torch.stack(neg_list).to(self.device)  # (B, num_negs, T, 52)

        return {"anchor": anchor, "positive": positive, "negatives": negatives}

    def train_one_epoch(self, epoch_idx: int = 0) -> dict:
        """跑 1 epoch 训练，返回 logs 摘要."""
        # 把所有视频切成窗口 pool
        all_windows = []
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            windows = split_to_windows(sample, seq_len=self.cfg.seq_len, stride=10)
            all_windows.extend(windows)
        print(f"  total windows: {len(all_windows)} from {len(self.dataset)} videos")

        # DataLoader + per-epoch reseed（PC-2）
        windows_ds = WindowDataset(all_windows)
        loader = DataLoader(
            windows_ds,
            batch_size=self.cfg.batch_size if self.cfg.use_ode_loss else self.cfg.batch_size * 4,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.cfg.seed + epoch_idx),
            collate_fn=lambda x: x,
        )

        steps_per_epoch = max(len(all_windows) // self.cfg.batch_size, 10)
        t0 = time.time()
        tau_start = self._collect_tau_values()
        tau_grad_buffer: list[float] = []
        first_batch_vids: Optional[list[str]] = None
        processed_steps = 0
        epoch_step_losses: list[float] = []

        for step_idx, sampled_windows in enumerate(loader):
            if step_idx >= steps_per_epoch:
                break
            self.step = step_idx
            processed_steps += 1
            self._freeze_f_net_if_needed()

            # 诊断：epoch 间第一个 batch 的视频序列必须变化
            if first_batch_vids is None:
                first_batch_vids = [str(w["video_id"]) for w in sampled_windows]
                print(f"  epoch {epoch_idx} first batch vids: {first_batch_vids[:12]}")
                if self._last_first_batch_vids is not None:
                    diff = sum(
                        a != b
                        for a, b in zip(self._last_first_batch_vids, first_batch_vids)
                    )
                    overlap = min(len(self._last_first_batch_vids), len(first_batch_vids))
                    if overlap > 0:
                        print(
                            f"  epoch {epoch_idx-1} vs {epoch_idx} first-batch diff={diff}/{overlap}",
                            flush=True,
                        )

            metric_terms: dict[str, float] = {}
            if self.cfg.use_ode_loss:
                # ADR-023: 直接整段时序输入，不走 contrastive pair 构造
                x = torch.stack([w["event_tensor"] for w in sampled_windows]).to(self.device)
                B = x.size(0)
                persona_idx = torch.zeros(B, dtype=torch.long, device=self.device)
                h_seq = self.encode(x, persona_idx)
                h_a = h_seq[:, -1, :]
                h_p = h_seq[:, -2, :] if h_seq.size(1) > 1 else h_seq[:, -1, :]
                ode_out = self.ode_loss(
                    x=x,
                    h_seq=h_seq,
                    backbone_step_fn=self._backbone_step_for_ode,
                    persona_idx=persona_idx,
                )
                loss = ode_out["loss"]
                metric_terms = {
                    "L_predict": float(ode_out["L_predict"].item()),
                    "L_recon": float(ode_out["L_recon"].item()),
                    "L_ode": float(ode_out["L_ode"].item()),
                    "L_infonce_aux": float(ode_out["L_infonce_aux"].item()),
                }
            else:
                pairs = self._build_contrastive_pairs(sampled_windows)
                if pairs is None:
                    continue

                anchor, positive, negatives = pairs["anchor"], pairs["positive"], pairs["negatives"]
                B = anchor.size(0)

                # 硬门 #3: AU augmentation on anchor + positive + negatives
                # 破坏 video-static 特征，强迫学时序结构（gemini 钦定）
                if self.cfg.au_augment_enabled:
                    aug_cfg = AUAugmentConfig(
                        channel_dropout_p=self.cfg.au_dropout_p,
                        intensity_jitter_std=self.cfg.au_jitter_std,
                        temporal_mixup_alpha=self.cfg.au_temporal_mixup_alpha,
                        frame_independent=True,  # gemini 钦定 non-coherent
                    )
                    anchor = au_augment(anchor, aug_cfg, seed=self.step * 1000 + 0)
                    positive = au_augment(positive, aug_cfg, seed=self.step * 1000 + 1)
                    # negatives shape (B, K, T, D), flatten 处理
                    B_n, K_n, T_n, D_n = negatives.shape
                    neg_flat = au_augment(
                        negatives.view(B_n * K_n, T_n, D_n),
                        aug_cfg,
                        seed=self.step * 1000 + 2,
                    )
                    negatives = neg_flat.view(B_n, K_n, T_n, D_n)

                # persona_idx: spike 阶段简化全 0
                persona_idx = torch.zeros(B, dtype=torch.long, device=self.device)

                # Forward
                h_a = self.encode(anchor, persona_idx)[:, -1, :]  # (B, 64) 取最后帧
                h_p = self.encode(positive, persona_idx)[:, -1, :]
                # negatives shape: (B, num_negs, T, 52) → flatten 跑 → reshape
                B_, K, T, D = negatives.shape
                persona_idx_n = torch.zeros(B_ * K, dtype=torch.long, device=self.device)
                h_n_flat = self.encode(negatives.view(B_ * K, T, D), persona_idx_n)[:, -1, :]
                h_n = h_n_flat.view(B_, K, -1)  # (B, num_negs, 64)

                # Loss
                if self.cfg.use_vicreg:
                    loss_dict = self.loss_fn(h_a, h_p)
                    loss = loss_dict["loss"]
                else:
                    # 时序突变 mask（gemini F-5 闭环）
                    breakpoint_mask = None
                    if self.cfg.use_breakpoint_mask:
                        bp = detect_semantic_breakpoints(anchor, sigma_threshold=2.0)
                        breakpoint_mask = bp.any(dim=1)  # (B,) 任一突变即 mask 该样本
                    loss = self.loss_fn(h_a, h_p, h_n, breakpoint_mask=breakpoint_mask)
            # PC-3 v3 解耦修法（gemini 第 8b + codex 第 8b 协同钦定 2026-05-22）：
            # - std_loss **常开**（防方差归零作弊，不受 λ 控制）
            # - cov_loss 受 λ 软门控（按 effective_dim 自适应）
            # 前因：v2 把 std+cov 整体乘 λ，λ<0.01 → std 也被压死 → 50ep effective_dim 3.92→坍塌
            vicreg_cfg = self.vicreg_trigger.cfg
            if (not self.cfg.use_ode_loss) and vicreg_cfg.enabled:
                std_loss = self.vicreg_trigger.compute_std_loss(h_a, h_p)
                loss = loss + vicreg_cfg.std_weight * std_loss
                lam = self.vicreg_trigger.current_lambda
                if lam > 1e-3:
                    cov_loss = self.vicreg_trigger.compute_cov_loss(h_a, h_p)
                    loss = loss + lam * vicreg_cfg.cov_weight * cov_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._capture_tau_grad(tau_grad_buffer)
            epoch_step_losses.append(float(loss.item()))

            # 监控 bias_grad / f_net_grad 比例（gemini F-4.1 + ADR-022）
            ratio = self._compute_grad_ratio()

            # ADR-022 DGB: 强制压制 τ-bias 梯度比例
            if self.cfg.dgb_enabled:
                self._apply_dgb()
                # 重新算 ratio 看 DGB 效果
                ratio_after_dgb = self._compute_grad_ratio()
            else:
                ratio_after_dgb = ratio

            self.optimizer.step()

            # collapse 监控（每 N step）
            if self.step % self.cfg.log_every_n == 0:
                with torch.no_grad():
                    metrics = self.monitor.step(h_a, {})
                elapsed = time.time() - t0
                log_entry = {
                    "step": self.step,
                    "loss": loss.item(),
                    "bias_grad_ratio_pre_dgb": ratio,
                    "bias_grad_ratio_post_dgb": ratio_after_dgb,
                    "collapse_severity": metrics["collapse_severity"],
                    "hidden_var": metrics["var"],
                    "pairwise_dist": metrics["pairwise_dist"],
                    "elapsed_s": elapsed,
                }
                log_entry.update(metric_terms)
                self.logs.append(log_entry)
                print(f"  step={self.step:3d} loss={loss.item():.4f} "
                      f"ratio={ratio:.2e}→DGB→{ratio_after_dgb:.2e} "
                      f"collapse={metrics['collapse_severity']} "
                      f"var={metrics['var']:.4f}",
                      flush=True)
                # ADR-022 health gate
                if not (0.1 <= ratio_after_dgb <= 50.0):
                    print(f"  ⚠️ HEALTH GATE post-DGB ratio={ratio_after_dgb:.2e} "
                          f"∉ [0.1, 50] (ADR-022 §2.3)", flush=True)
                if ratio < 1e-3 and not self._freeze_f_net_if_needed_check():
                    print(f"  ⚠️ ALARM pre-DGB ratio < 1e-3 (gemini F-4.1 shortcut)", flush=True)

        self._last_first_batch_vids = first_batch_vids
        tau_end = self._collect_tau_values()
        tau_stats = self._summarize_tau_stats(tau_end, tau_start, tau_grad_buffer)
        print(
            f"  τ mean={tau_stats['mean']:.3f} ± {tau_stats['std']:.3f} "
            f"∈ [{tau_stats['min']:.3f}, {tau_stats['max']:.3f}] "
            f"grad={tau_stats['grad_norm']:.2e}",
            flush=True,
        )

        epoch_log = {
            "epoch": epoch_idx,
            "loss_mean": float(sum(epoch_step_losses) / len(epoch_step_losses))
            if epoch_step_losses
            else None,
            "steps": processed_steps,
            "tau_stats": tau_stats,
            "first_batch_vids": first_batch_vids,
        }
        self.epoch_logs.append(epoch_log)

        return {
            "total_steps": processed_steps,
            "final_loss": self.logs[-1]["loss"] if self.logs else None,
            "logs": self.logs,
            "epoch_log": epoch_log,
            "tau_stats": tau_stats,
            "elapsed_s": time.time() - t0,
        }

    def on_epoch_end_with_eval(self, eval_result: dict[str, Any], epoch_idx: int) -> dict[str, Any]:
        """PC-3: evaluate 后更新 VICReg auto-trigger 状态并回填 epoch log."""
        newly = self.vicreg_trigger.update(
            effective_dim=eval_result["effective_dim"],
            epoch=epoch_idx,
        )
        if newly:
            print(
                f"⚠️  VICReg Std+Cov AUTO-TRIGGERED "
                f"(effective_dim={eval_result['effective_dim']:.2f})"
            )
        trigger_log = self.vicreg_trigger.to_log()
        if self.epoch_logs and self.epoch_logs[-1].get("epoch") == epoch_idx:
            self.epoch_logs[-1].update(trigger_log)
        return trigger_log

    def _freeze_f_net_if_needed_check(self) -> bool:
        return self.step < self.cfg.freeze_f_net_steps

    def _collect_tau_values(self) -> torch.Tensor:
        """Collect current τ values from all CfC cells for monitoring."""
        values = []
        with torch.no_grad():
            persona_idx = torch.zeros(1, dtype=torch.long, device=self.device)
            for cell in self.backbone.cells:
                tau = cell.compute_tau(persona_idx).reshape(-1)
                values.append(tau.detach().cpu())
        return torch.cat(values) if values else torch.zeros(1)

    def _capture_tau_grad(self, tau_grad_buffer: list[float]) -> None:
        """Capture τ.grad right after backward and before optimizer step/zero_grad."""
        for p in self.persona_bias_params:
            if p.grad is not None:
                tau_grad_buffer.append(float(p.grad.detach().norm().item()))

    @staticmethod
    def _summarize_tau_stats(
        tau_values: torch.Tensor,
        tau_values_start: torch.Tensor,
        tau_grad_buffer: list[float],
    ) -> dict[str, float]:
        grad_mean = float(sum(tau_grad_buffer) / len(tau_grad_buffer)) if tau_grad_buffer else 0.0
        mean_abs_delta = float((tau_values - tau_values_start).abs().mean().item())
        return {
            "mean": float(tau_values.mean().item()),
            "min": float(tau_values.min().item()),
            "max": float(tau_values.max().item()),
            "std": float(tau_values.std(unbiased=False).item()),
            "grad_norm": grad_mean,
            "mean_abs_delta": mean_abs_delta,
        }

    def _compute_grad_ratio(self) -> float:
        bias_norm, f_net_norm = 0.0, 0.0
        for cell in self.backbone.cells:
            for n, p in cell.named_parameters():
                if p.grad is None:
                    continue
                if "persona_tau_bias" in n:
                    bias_norm += p.grad.abs().sum().item()
                elif "f_net" in n:
                    f_net_norm += p.grad.abs().sum().item()
        return bias_norm / max(f_net_norm, 1e-9)

    def _apply_dgb(self) -> None:
        """ADR-022 §2.2: Dynamic Gradient Balancing
        强制 bias_grad/f_net_grad ≤ 1/target_ratio（默认 1/10）
        只缩小 τ-bias 梯度，不放大（避免反向梯度爆炸）
        """
        bias_norm = 0.0
        f_net_norm = 0.0
        for p in self.persona_bias_params:
            if p.grad is not None:
                bias_norm += p.grad.norm().item() ** 2
        for p in self.f_net_params:
            if p.grad is not None:
                f_net_norm += p.grad.norm().item() ** 2
        bias_norm = max(bias_norm ** 0.5, 1e-9)
        f_net_norm = max(f_net_norm ** 0.5, 1e-9)

        # 目标：bias_norm ≤ f_net_norm / target_ratio
        max_allowed = f_net_norm / self.cfg.dgb_target_ratio
        if bias_norm > max_allowed:
            scale = max_allowed / bias_norm
            for p in self.persona_bias_params:
                if p.grad is not None:
                    p.grad.data *= scale

    def save_checkpoint(self, path: str | Path):
        """save fusion + backbone state（≤ 20MB spec §1.2）"""
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "fusion_state": self.fusion.state_dict(),
            "backbone_state": self.backbone.state_dict(),
            "cfg": dataclasses.asdict(self.cfg),
            "logs": self.logs,
            "step": self.step,
        }, path)
        size_mb = path.stat().st_size / 1e6
        print(f"✅ ckpt saved {path} ({size_mb:.2f} MB)")
        assert size_mb < 20, f"ckpt size {size_mb}MB > spec 20MB 上限"
