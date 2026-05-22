"""v3.0 checkpoint inference driver for daemon AU cache."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .daemon_dataset import DaemonAUDataset
from .evaluate import measure_input_effective_dim, run_evaluation
from .trainer import TinyTrainer, TrainConfig


def _build_cfg(
    raw_cfg: dict[str, Any] | None,
    seq_len: int,
    fusion_state: dict[str, Any] | None = None,
    backbone_state: dict[str, Any] | None = None,
) -> TrainConfig:
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    cfg_dict = dict(raw_cfg or {})
    if fusion_state and "hidden_dim" not in cfg_dict and "proj.weight" in fusion_state:
        cfg_dict["hidden_dim"] = int(fusion_state["proj.weight"].shape[0])
    if backbone_state:
        if "num_personas" not in cfg_dict:
            for key, val in backbone_state.items():
                if key.endswith("persona_tau_bias.weight") and hasattr(val, "shape"):
                    cfg_dict["num_personas"] = int(val.shape[0])
                    break
        if "num_layers" not in cfg_dict:
            layer_ids: list[int] = []
            for key in backbone_state:
                m = re.match(r"cells\.(\d+)\.", key)
                if m:
                    layer_ids.append(int(m.group(1)))
            if layer_ids:
                cfg_dict["num_layers"] = max(layer_ids) + 1
    cfg_dict["seq_len"] = int(cfg_dict.get("seq_len", seq_len))
    cfg_dict["use_ode_loss"] = True
    filtered = {k: v for k, v in cfg_dict.items() if k in fields}
    return TrainConfig(**filtered)


def _first_state(ckpt: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any] | None:
    for key in keys:
        value = ckpt.get(key)
        if isinstance(value, dict):
            return value
    return None


def _dummy_dataset() -> DaemonAUDataset:
    tmp = Path("/tmp/lnmem_infer_dummy_dataset.npz")
    if not tmp.exists():
        np.savez_compressed(tmp, au_dummy=np.zeros((30, 20), dtype=np.float32))
    return DaemonAUDataset(tmp)


def load_v30_checkpoint(ckpt_path: Path, device: str = "cpu") -> TinyTrainer:
    """Restore TinyTrainer from .pt checkpoint. / 从 checkpoint 还原训练器。"""
    payload = torch.load(Path(ckpt_path), map_location=torch.device(device), weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(payload)!r}")

    fusion_state = _first_state(payload, ("fusion_state", "fusion", "state_fusion"))
    backbone_state = _first_state(payload, ("backbone_state", "backbone", "state_backbone"))
    decoder_state = _first_state(payload, ("decoder_state", "decoder"))

    if fusion_state is None or backbone_state is None:
        raise KeyError("checkpoint missing fusion/backbone state dict")

    cfg = _build_cfg(
        payload.get("cfg"),
        seq_len=30,
        fusion_state=fusion_state,
        backbone_state=backbone_state,
    )
    trainer = TinyTrainer(cfg=cfg, dataset=_dummy_dataset(), device=device)

    trainer.fusion.load_state_dict(fusion_state)
    trainer.backbone.load_state_dict(backbone_state)
    if decoder_state is not None:
        trainer.decoder.load_state_dict(decoder_state, strict=False)

    trainer.step = int(payload.get("step", 0))
    trainer.fusion.eval()
    trainer.backbone.eval()
    trainer.decoder.eval()
    return trainer


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, torch.Tensor):
        return _jsonable(v.detach().cpu().tolist())
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float):
        return None if not math.isfinite(v) else v
    return v


def _cache_stats(dataset: DaemonAUDataset) -> tuple[int, int]:
    n_sessions = len(dataset.video_ids)
    total_frames = int(sum(arr.shape[0] for arr in dataset.au_cache.values()))
    return n_sessions, total_frames


def run_inference(
    ckpt_path: Path,
    cache_path: Path,
    out_path: Path,
    sample_per_video: int = 5,
    seq_len: int = 30,
) -> dict[str, Any]:
    """Run ckpt + daemon cache evaluation and write result JSON. / 串联评估并落盘。"""
    t0 = time.time()
    dataset = DaemonAUDataset(cache_path)
    trainer = load_v30_checkpoint(ckpt_path=ckpt_path, device="cpu")
    trainer.dataset = dataset
    trainer.cfg = dataclasses.replace(trainer.cfg, seq_len=seq_len, use_ode_loss=True)

    eval_result = run_evaluation(
        trainer=trainer,
        dataset=dataset,
        sample_per_video=sample_per_video,
        include_shuffle_test=True,
        spike_phase=True,
    )

    input_effective_dim = float(measure_input_effective_dim(dataset))
    output_effective_dim = float(eval_result.get("effective_dim", float("nan")))
    dim_ratio = output_effective_dim / input_effective_dim if input_effective_dim > 1e-9 else float("nan")

    n_sessions, total_frames = _cache_stats(dataset)
    gates_summary = {
        "gate_1_tau_std": "N/A (需 trainer 内部状态)",
        "gate_2_vs_persistence": float(eval_result.get("model_vs_persistence_ratio", float("nan"))),
        "gate_3_effective_dim_ratio": float(dim_ratio),
        "gate_4_shuffle_ratio": float(eval_result.get("shuffle_increase_ratio", float("nan"))),
        "gate_5_trivial_recon_check": bool(eval_result.get("trivial_recon_check_pass", False)),
        "gate_6_rare_au_verdict": str(eval_result.get("rare_au_verdict", "N/A")),
    }

    out = {
        "ckpt_path": str(ckpt_path),
        "cache_path": str(cache_path),
        "n_sessions": int(n_sessions),
        "total_frames": int(total_frames),
        "eval_result": eval_result,
        "elapsed_s": round(time.time() - t0, 3),
        "gates_summary": gates_summary,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonable(out), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    """CLI entrypoint. / 命令行入口。"""
    parser = argparse.ArgumentParser(description="Run v3.0 ckpt evaluation on daemon AU cache")
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--daemon-cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sample-per-video", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=30)
    args = parser.parse_args()

    result = run_inference(
        ckpt_path=args.ckpt,
        cache_path=args.daemon_cache,
        out_path=args.out,
        sample_per_video=args.sample_per_video,
        seq_len=args.seq_len,
    )
    print(json.dumps(_jsonable(result["gates_summary"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
