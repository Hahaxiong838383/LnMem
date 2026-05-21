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
import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from poc.lnn import MultiLayerCustomCfC
from poc.lnn.heads import FusionLayer, FusionConfig
from poc.lnn.losses import InfoNCELoss, InfoNCEConfig, VICRegLoss, VICRegConfig, detect_semantic_breakpoints
from poc.monitoring.collapse_monitor import CollapseMonitor
from .dataset import VEATICDataset, split_to_windows


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
    # ADR-022: 双率优化器 + DGB
    tau_lr: float = 1e-6                # τ-bias 极低 LR（vs 主 lr 1e-3 差 1000x）
    dgb_target_ratio: float = 10.0      # DGB 强制 bias_grad/f_net_grad ≤ 1/10
    dgb_enabled: bool = True
    persona_lr_scale: float = 1.0       # DEPRECATED by ADR-022 双率优化器
    f_net_weight_decay: float = 1e-3    # gemini F-4.1
    freeze_f_net_steps: int = 200       # spike 简化（spec 1k step）
    use_vicreg: bool = False            # False = InfoNCE 主路径
    use_breakpoint_mask: bool = True    # gemini 终审条件 1
    log_every_n: int = 50
    seed: int = 42


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

        # collapse 监控
        self.monitor = CollapseMonitor()

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

        # ADR-022: 双率优化器（τ-bias 用极低 LR 1e-6，主干 1e-3）
        self.persona_bias_params = persona_bias_params
        self.f_net_params = f_net_params
        self.optimizer = AdamW([
            {"params": other_params, "lr": cfg.lr, "weight_decay": 0.0},
            {"params": f_net_params, "lr": cfg.lr, "weight_decay": cfg.f_net_weight_decay},
            {"params": persona_bias_params, "lr": cfg.tau_lr, "weight_decay": 0.0},
        ])

        self.step = 0
        self.logs = []

    def encode(self, x: torch.Tensor, persona_idx: torch.Tensor) -> torch.Tensor:
        """52d → 64d hidden seq."""
        x_fused = self.fusion(x)  # (B, T, 64)
        h_seq = self.backbone(x_fused, persona_idx)  # (B, T, 64)
        return h_seq

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

    def train_one_epoch(self) -> dict:
        """跑 1 epoch 训练，返回 logs 摘要."""
        # 把所有视频切成窗口 pool
        all_windows = []
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            windows = split_to_windows(sample, seq_len=self.cfg.seq_len, stride=10)
            all_windows.extend(windows)
        print(f"  total windows: {len(all_windows)} from {len(self.dataset)} videos")

        # 每个 step 从 pool 随机 sample 一个 batch
        steps_per_epoch = max(len(all_windows) // self.cfg.batch_size, 10)
        t0 = time.time()

        for self.step in range(steps_per_epoch):
            self._freeze_f_net_if_needed()

            # 抽 batch（取 batch_size + 一些备选构造 contrastive pairs）
            indices = torch.randperm(len(all_windows))[:self.cfg.batch_size * 4]
            sampled_windows = [all_windows[i] for i in indices]
            pairs = self._build_contrastive_pairs(sampled_windows)
            if pairs is None:
                continue

            anchor, positive, negatives = pairs["anchor"], pairs["positive"], pairs["negatives"]
            B = anchor.size(0)

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

            self.optimizer.zero_grad()
            loss.backward()

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
                    "bias_grad_ratio": ratio,
                    "collapse_severity": metrics["collapse_severity"],
                    "hidden_var": metrics["var"],
                    "pairwise_dist": metrics["pairwise_dist"],
                    "elapsed_s": elapsed,
                }
                self.logs.append(log_entry)
                print(f"  step={self.step:3d} loss={loss.item():.4f} "
                      f"bias_grad_ratio={ratio:.6f} "
                      f"collapse={metrics['collapse_severity']} "
                      f"var={metrics['var']:.4f} elapsed={elapsed:.1f}s",
                      flush=True)
                if ratio < 1e-3 and not self._freeze_f_net_if_needed_check():
                    print(f"  ⚠️ ALARM bias_grad_ratio={ratio:.6e} < 1e-3 "
                          f"(gemini F-4.1 shortcut learning 警告)", flush=True)

        return {
            "total_steps": self.step + 1,
            "final_loss": self.logs[-1]["loss"] if self.logs else None,
            "logs": self.logs,
            "elapsed_s": time.time() - t0,
        }

    def _freeze_f_net_if_needed_check(self) -> bool:
        return self.step < self.cfg.freeze_f_net_steps

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
