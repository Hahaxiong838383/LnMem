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

from poc.lnn import (
    Decoder,
    MultiLayerCustomCfC,
    OdeDynamicsLoss,
    OdeDynamicsLossConfig,
    clamp_tau,
    invariance_loss,
)
from poc.lnn.heads import FusionLayer, FusionConfig
from poc.lnn.losses import InfoNCELoss, InfoNCEConfig, VICRegLoss, VICRegConfig, detect_semantic_breakpoints
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
    decoder_weight_decay: float = 1e-2  # M3 整改：限制 decoder 容量防 trivial 重构
    # R3 (gemini 12 + cc 13 数学纠正 + gemini 13 sign-off): τ-spread 正则强制不同通道 τ 跨数量级分化
    # gemini 12 钦定 Entropy(σ(τ)) 数学错（softmax+entropy 让 τ 均匀，反向；gemini 13 认错）
    # cc 改用 std(log τ_base) 直接衡量 τ 跨度，对齐 LNN 多时间尺度精神
    # gemini 13 建议 λ=0.05 起步（cc 原拍 0.1 太激进，防 τ 边界溢出）
    lambda_tau_spread: float = 0.05
    freeze_f_net_steps: int = 200       # spike 简化（spec 1k step）
    max_bptt_steps: int = 32            # gemini 第 7 轮 F-4 工程防御
    use_ode_loss: bool = True           # ADR-023 v2 主路径
    use_vicreg: bool = False            # False = InfoNCE 主路径
    use_breakpoint_mask: bool = True    # gemini 终审条件 1
    log_every_n: int = 50
    seed: int = 42
    # 硬门 #3 (gemini 第 5 轮): AU augmentation 替代 spatial aug
    au_augment_enabled: bool = True
    au_dropout_p: float = 0.3        # 实测甜区：M5 833→500（-40%）, Shuffle drop +0.241
    au_jitter_std: float = 0.05      # 同上
    au_temporal_mixup_alpha: float = 0.0  # 关：激进 mixup 反而劣化（实测 5/21）
    tau_upper_bound: float = 30.0
    invariance_weight: float = 0.1
    va_loss_weight: float = 0.0


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
        # R1 (gemini 第 12 轮承认 11 轮错钦定): 砍 bottleneck，回单层 Linear(64→52)
        # 论据: bottleneck=16 死死卡 effective_dim 上限，3.61 是简并必然结果
        # 跟 gemini 12 收敛: 释放表征通道，配合 R3 τ-spread 强迫真分化
        self.decoder = Decoder(hidden_dim=cfg.hidden_dim, input_dim=52, num_layers=1).to(self.device)
        self.va_head = torch.nn.Linear(cfg.hidden_dim, 2).to(self.device)
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
        for p in self.va_head.parameters():
            other_params.append(p)
        # M3 整改（cc 第 11 轮）：decoder 单独 group 加 wd=1e-2 限制 trivial 重构表达
        decoder_params: list = []
        if cfg.use_ode_loss:
            for p in self.decoder.parameters():
                decoder_params.append(p)

        # ADR-022 v3: 双率优化器（τ-bias 释放到 1e-3，主干由 DGB 主控约束）
        self.persona_bias_params = persona_bias_params
        self.f_net_params = f_net_params
        self.decoder_params = decoder_params
        param_groups = [
            {"params": other_params, "lr": cfg.lr, "weight_decay": 0.0},
            {"params": f_net_params, "lr": cfg.lr, "weight_decay": cfg.f_net_weight_decay},
            {"params": persona_bias_params, "lr": cfg.tau_lr, "weight_decay": 0.0},
        ]
        if decoder_params:
            # M3: 高 weight_decay 限制 decoder 容量，防 trivial 均值/瞬时重构
            param_groups.append({
                "params": decoder_params,
                "lr": cfg.lr,
                "weight_decay": cfg.decoder_weight_decay,
            })
        self.optimizer = AdamW(param_groups)

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

    def backbone_step(
        self,
        x_t: torch.Tensor,
        h_prev_list: list[torch.Tensor],
        persona_idx: torch.Tensor,
        dt: float = 1.0,
    ) -> list[torch.Tensor]:
        """Single-step fusion + CfC stack for rollout. / rollout 单步。"""
        x_t_fused = self.fusion(x_t.unsqueeze(1)).squeeze(1)
        return self.backbone.step(x_t_fused, h_prev_list, persona_idx, dt=dt)

    def _rollout_hidden_sequence(
        self,
        x: torch.Tensor,
        persona_idx: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Run teacher-forced hidden rollout for a full sequence."""
        h_states = [cell.init_hidden(x.size(0), x.device) for cell in self.backbone.cells]
        h_seq: list[torch.Tensor] = []
        for t in range(x.size(1)):
            h_states = self.backbone_step(x[:, t, :], h_states, persona_idx, dt=dt)
            h_seq.append(h_states[-1])
        return torch.stack(h_seq, dim=1)

    def _apply_tau_clamp(self, upper_bound: float) -> int:
        """Clamp tau_base upper bound for all cells and return changed count."""
        total = 0
        for cell in self.backbone.cells:
            total += clamp_tau(cell, upper_bound)
        return total

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

    def train_one_epoch(
        self,
        epoch_idx: int = 0,
        dataloader: DataLoader | None = None,
    ) -> dict:
        """跑 1 epoch 训练，返回 logs 摘要."""
        if dataloader is not None:
            return self._train_one_epoch_multifps(epoch_idx=epoch_idx, dataloader=dataloader)
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
        tau_clamp_count = 0
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
                x = torch.stack([w["event_tensor"] for w in sampled_windows]).to(self.device)
                batch_fps_set = {float(w.get("fps", 30.0)) for w in sampled_windows}
                if len(batch_fps_set) != 1:
                    raise ValueError(
                        f"mixed fps in one batch is not supported in phase-1: {sorted(batch_fps_set)}"
                    )
                batch_fps = next(iter(batch_fps_set))
                batch_size = x.size(0)
                persona_idx = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                ode_out = self.ode_loss(
                    x=x,
                    backbone_step_fn=self.backbone_step,
                    persona_idx=persona_idx,
                    init_hidden_fn=lambda b, dev: [
                        cell.init_hidden(b, dev) for cell in self.backbone.cells
                    ],
                    fps=batch_fps,
                )
                loss = ode_out["loss"]
                h_a = ode_out["h_seq_prefix"][:, -1, :]
                h_p = h_a
                metric_terms = {
                    "L_world_model": float(ode_out["L_world_model"].item()),
                    "L_recon_prefix": float(ode_out["L_recon_prefix"].item()),
                    "prefix_len": float(ode_out["prefix_len"]),
                    "rollout_len": float(ode_out["rollout_len"]),
                    "actual_dt": float(ode_out.get("actual_dt", 1.0)),
                }
            else:
                pairs = self._build_contrastive_pairs(sampled_windows)
                if pairs is None:
                    continue

                anchor, positive, negatives = pairs["anchor"], pairs["positive"], pairs["negatives"]
                batch_size = anchor.size(0)

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
                    b_n, k_n, t_n, d_n = negatives.shape
                    neg_flat = au_augment(
                        negatives.view(b_n * k_n, t_n, d_n),
                        aug_cfg,
                        seed=self.step * 1000 + 2,
                    )
                    negatives = neg_flat.view(b_n, k_n, t_n, d_n)

                # persona_idx: spike 阶段简化全 0
                persona_idx = torch.zeros(batch_size, dtype=torch.long, device=self.device)

                # Forward
                h_a = self.encode(anchor, persona_idx)[:, -1, :]  # (B, 64) 取最后帧
                h_p = self.encode(positive, persona_idx)[:, -1, :]
                # negatives shape: (B, num_negs, T, 52) → flatten 跑 → reshape
                b_n, k_n, t_n, d_n = negatives.shape
                persona_idx_n = torch.zeros(b_n * k_n, dtype=torch.long, device=self.device)
                h_n_flat = self.encode(negatives.view(b_n * k_n, t_n, d_n), persona_idx_n)[:, -1, :]
                h_n = h_n_flat.view(b_n, k_n, -1)  # (B, num_negs, 64)

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

            # R3 (gemini 12 + cc 13): τ-spread 正则强制多时间尺度
            if self.cfg.lambda_tau_spread > 0:
                tau_spread_loss = self._compute_tau_spread_loss()
                loss = loss + self.cfg.lambda_tau_spread * tau_spread_loss

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
            tau_clamp_count += self._apply_tau_clamp(self.cfg.tau_upper_bound)

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
            "tau_clamp_count": tau_clamp_count,
        }
        self.epoch_logs.append(epoch_log)

        return {
            "total_steps": processed_steps,
            "final_loss": self.logs[-1]["loss"] if self.logs else None,
            "logs": self.logs,
            "epoch_log": epoch_log,
            "tau_stats": tau_stats,
            "tau_clamp_count": tau_clamp_count,
            "elapsed_s": time.time() - t0,
        }

    def _train_one_epoch_multifps(self, epoch_idx: int, dataloader: DataLoader) -> dict:
        """Multi-FPS curriculum training loop with invariance + tau clamp."""
        sampler = getattr(dataloader, "batch_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch_idx)

        t0 = time.time()
        tau_start = self._collect_tau_values()
        tau_grad_buffer: list[float] = []
        epoch_step_losses: list[float] = []
        processed_steps = 0
        tau_clamp_count = 0
        per_fps_loss: dict[float, list[float]] = {}

        for step_idx, sampled_windows in enumerate(dataloader):
            if not sampled_windows:
                continue
            self.step = step_idx
            processed_steps += 1
            self._freeze_f_net_if_needed()

            x = torch.stack([w["event_tensor"] for w in sampled_windows]).to(self.device)
            labels_va = torch.stack([w["labels_va"] for w in sampled_windows]).to(self.device)
            batch_fps_set = {float(w.get("fps", 30.0)) for w in sampled_windows}
            if len(batch_fps_set) != 1:
                raise ValueError(f"mixed fps in one batch is not supported: {sorted(batch_fps_set)}")
            batch_fps = next(iter(batch_fps_set))

            batch_size = x.size(0)
            persona_idx = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            ode_out = self.ode_loss(
                x=x,
                backbone_step_fn=self.backbone_step,
                persona_idx=persona_idx,
                init_hidden_fn=lambda b, dev: [cell.init_hidden(b, dev) for cell in self.backbone.cells],
                fps=batch_fps,
            )
            loss = ode_out["loss"]

            # F1: Conditional V-A loss masking when labels are missing (daemon=all zeros).
            va_loss = torch.tensor(0.0, device=self.device)
            va_mask_count = int((labels_va.abs().sum(dim=(1, 2)) > 0).sum().item())
            if self.cfg.va_loss_weight > 0 and va_mask_count > 0:
                prefix_len = int(ode_out["prefix_len"])
                va_pred = self.va_head(ode_out["h_seq_prefix"])
                va_target = labels_va[:, :prefix_len, :]
                valid = va_target.abs().sum(dim=(1, 2)) > 0
                va_loss = torch.nn.functional.mse_loss(va_pred[valid], va_target[valid])
                loss = loss + self.cfg.va_loss_weight * va_loss

            # F1: Invariance loss aligns high-fps hidden with low-fps subsampled hidden.
            # cc 修补 (5/23)：原条件 `x.size(1) >= fps_ratio*2` 在 seq_len=8 + fps_ratio=30 时永不触发 → invariance 失效。
            # 改为：fps_ratio 自适应到当前 seq_len，至少 2 帧对齐即可（数学有效性最小条件）。
            inv_loss = torch.tensor(0.0, device=self.device)
            if self.cfg.invariance_weight > 0 and batch_fps >= 2.0:
                max_ratio = max(1, x.size(1) // 2)
                fps_ratio = min(max(1, int(round(batch_fps))), max_ratio)
                if fps_ratio >= 2:
                    x_low = x[:, ::fps_ratio, :]
                    low_fps = batch_fps / fps_ratio  # 实际 low fps 由 ratio 反推
                    h_high = self._rollout_hidden_sequence(x, persona_idx, dt=1.0 / batch_fps)
                    h_low = self._rollout_hidden_sequence(x_low, persona_idx, dt=1.0 / max(low_fps, 1e-6))
                    inv_loss = invariance_loss(h_high, h_low, fps_ratio=fps_ratio)
                    loss = loss + self.cfg.invariance_weight * inv_loss

            if self.cfg.lambda_tau_spread > 0:
                tau_spread_loss = self._compute_tau_spread_loss()
                loss = loss + self.cfg.lambda_tau_spread * tau_spread_loss
            else:
                tau_spread_loss = torch.tensor(0.0, device=self.device)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._capture_tau_grad(tau_grad_buffer)
            epoch_step_losses.append(float(loss.item()))

            ratio = self._compute_grad_ratio()
            if self.cfg.dgb_enabled:
                self._apply_dgb()
                ratio_after_dgb = self._compute_grad_ratio()
            else:
                ratio_after_dgb = ratio

            self.optimizer.step()
            tau_clamp_count += self._apply_tau_clamp(self.cfg.tau_upper_bound)

            per_fps_loss.setdefault(float(batch_fps), []).append(float(ode_out["L_world_model"].item()))

            if self.step % self.cfg.log_every_n == 0:
                log_entry = {
                    "step": self.step,
                    "loss": float(loss.item()),
                    "L_world_model": float(ode_out["L_world_model"].item()),
                    "L_recon_prefix": float(ode_out["L_recon_prefix"].item()),
                    "L_invariance": float(inv_loss.item()),
                    "L_va": float(va_loss.item()),
                    "va_mask_count": va_mask_count,
                    "tau_spread_loss": float(tau_spread_loss.item()),
                    "fps": float(batch_fps),
                    "actual_dt": float(ode_out.get("actual_dt", 1.0)),
                    "bias_grad_ratio_pre_dgb": ratio,
                    "bias_grad_ratio_post_dgb": ratio_after_dgb,
                }
                self.logs.append(log_entry)
                print(
                    f"  step={self.step:3d} fps={batch_fps:.2f} "
                    f"loss={loss.item():.4f} inv={inv_loss.item():.4f} "
                    f"tau_clamp={tau_clamp_count}",
                    flush=True,
                )

        tau_end = self._collect_tau_values()
        tau_stats = self._summarize_tau_stats(tau_end, tau_start, tau_grad_buffer)
        per_fps_mean = {k: float(sum(v) / len(v)) for k, v in sorted(per_fps_loss.items()) if v}
        epoch_log = {
            "epoch": epoch_idx,
            "loss_mean": float(sum(epoch_step_losses) / len(epoch_step_losses))
            if epoch_step_losses
            else None,
            "steps": processed_steps,
            "tau_stats": tau_stats,
            "tau_clamp_count": tau_clamp_count,
            "per_fps_loss": per_fps_mean,
        }
        self.epoch_logs.append(epoch_log)
        return {
            "total_steps": processed_steps,
            "final_loss": self.logs[-1]["loss"] if self.logs else None,
            "logs": self.logs,
            "epoch_log": epoch_log,
            "tau_stats": tau_stats,
            "tau_clamp_count": tau_clamp_count,
            "per_fps_loss": per_fps_mean,
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

    def _compute_tau_spread_loss(self) -> torch.Tensor:
        """R3 (gemini 12 + cc 13 数学纠正): τ-spread 正则强制不同通道 τ 跨数量级分化.

        数学论据（cc 改 gemini 12 的 Entropy(σ(τ)) 公式）：
        - gemini 12 公式: Entropy(softmax(τ)) 最大化 → τ 分布均匀（所有 τ 接近相等），
          跟"强制分化"的意图反。CLAUDE.md §3.1 cc 与 gemini 冲突时必明示分歧。
        - cc 公式: std(log τ_base) 最大化 → τ 跨数量级分化（毫秒→秒→分钟），
          对齐 LNN 多时间尺度精神（Hasani et al. 2021）+ ADR-022 v3 τ 释放自然延续。

        Returns:
            scalar loss = -std(log τ_base)，最小化它 = 最大化 spread
        """
        log_taus = []
        for cell in self.backbone.cells:
            log_taus.append(torch.log(cell.tau_base.clamp(min=1e-6)))
        log_tau = torch.cat(log_taus)  # (num_layers * hidden_dim,)
        if log_tau.numel() < 2:
            return torch.tensor(0.0, device=log_tau.device, requires_grad=True)
        spread = log_tau.std(unbiased=False)
        return -spread

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
            "va_head_state": self.va_head.state_dict(),
            "cfg": dataclasses.asdict(self.cfg),
            "logs": self.logs,
            "step": self.step,
        }, path)
        size_mb = path.stat().st_size / 1e6
        print(f"✅ ckpt saved {path} ({size_mb:.2f} MB)")
        assert size_mb < 20, f"ckpt size {size_mb}MB > spec 20MB 上限"
