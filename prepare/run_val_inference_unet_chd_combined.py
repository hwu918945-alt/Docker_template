#!/usr/bin/env python3
"""Run CHD UNet segmentation (train_unet_chd.py preprocessing) + DINOv3 FiLM classifier."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
_sys_p = Path(__file__).resolve().parent
if str(_sys_p) not in sys.path:
    sys.path.insert(0, str(_sys_p))
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

logger = logging.getLogger("val_infer_unet_chd")

DEFAULT_SEG_CKPT = Path("/ws/prepare/best.pth")
DEFAULT_CLS_SCRIPT = Path("/ws/prepare/infer_full_pos7_hist_rel_v2_revise_vote_v2.py")
DEFAULT_CLS_CKPT = Path("/ws/prepare/best_fc.pt")
DEFAULT_CLS_WEIGHTS = Path("/ws/prepare/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
DEFAULT_CLS_DISEASE_JSON = Path("/ws/prepare/disease_cases.json")
DEFAULT_CLS_THRESHOLDS = "0.5,0.5,0.5,0.5,0.5,0.5,0.5"


@dataclass
class SegConfig:
    img_size: int
    patch_size: Optional[Tuple[int, int]]
    normalize: str
    mean: Optional[Tuple[float, float, float]]
    std: Optional[Tuple[float, float, float]]
    num_classes: int
    in_channels: int
    base_channels: int
    bilinear: bool
    nnunet_plan: Optional[Path]
    nnunet_plan_config: str
    plan_override: bool
    use_plan_arch: bool


def torch_load_ckpt(path: Path, map_location: str | torch.device):
    """Load a checkpoint compatibly across torch versions (PyTorch 2.6 defaults weights_only=True)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_args():
    p = argparse.ArgumentParser(description="Run CHD UNet + DINOv3 FiLM on a H5 set.")
    p.add_argument("--images", type=Path, default=None, help="Directory with {case_id}.h5")
    p.add_argument("--val-images", type=Path, default=None, help="Alias for --images")
    p.add_argument("--json", type=Path, default=None, help="Optional JSON list of case ids")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for combined predictions")
    p.add_argument("--tmp-dir", type=Path, default=Path("tmp/val_infer_unet"))
    p.add_argument("--keep-temp", action="store_true")
    p.add_argument("--preview-num", type=int, default=0, help="Run a small preview on N cases before full inference.")
    p.add_argument("--preview-out", type=Path, default=Path("preview_vis"), help="Preview visualization output dir.")
    p.add_argument("--preview-seed", type=int, default=42)

    # segmentation (UNet)
    p.add_argument("--seg-ckpt", type=Path, default=DEFAULT_SEG_CKPT)
    p.add_argument("--seg-device", type=str, default=None)
    p.add_argument("--seg-batch-size", type=int, default=4)
    p.add_argument("--seg-num-workers", type=int, default=1)

    # classifier (DINOv3 FiLM)
    p.add_argument("--cls-script", type=Path, default=DEFAULT_CLS_SCRIPT)
    p.add_argument("--cls-ckpt", type=Path, default=DEFAULT_CLS_CKPT)
    p.add_argument("--cls-weights", type=Path, default=DEFAULT_CLS_WEIGHTS)
    p.add_argument("--cls-disease-json", type=Path, default=DEFAULT_CLS_DISEASE_JSON)
    p.add_argument("--cls-img-size", type=int, default=224)
    p.add_argument("--cls-patch-size", type=int, default=224)
    p.add_argument("--cls-locals-total", type=int, default=8)
    p.add_argument("--cls-locals-greedy", type=int, default=8)
    p.add_argument("--cls-min-dist-ratio", type=float, default=0.5)
    p.add_argument("--cls-min-overlap-ratio", type=float, default=0.1)
    p.add_argument("--cls-bbox-margin", type=float, default=0.15)
    p.add_argument("--cls-erode-kernel", type=int, default=5)
    p.add_argument("--cls-erode-iters", type=int, default=1)
    p.add_argument("--cls-film-hidden", type=int, default=64)
    p.add_argument("--cls-film-scale", type=float, default=0.02)
    p.add_argument("--cls-film-scale-pos", type=float, default=0.02)
    p.add_argument("--cls-film-scale-mask", type=float, default=0.005)
    p.add_argument("--cls-mask-cond", type=str, default="hist")
    p.add_argument("--cls-mask-cond-dropout", type=float, default=0.5)
    p.add_argument("--cls-mask-hist-ids", type=str, default="1,2,3,4,5,6,7,8,9,12,13")
    p.add_argument("--cls-mask-hist-patch-size", type=int, default=64)
    p.add_argument("--cls-local-total-weight", type=float, default=0.8)
    p.add_argument("--cls-agg-mode", type=str, default="topk", choices=["sum", "topk"])
    p.add_argument("--cls-topk", type=int, default=6)
    p.add_argument("--cls-local-scale", type=float, default=-1.0)
    p.add_argument("--cls-global-scale", type=float, default=0.2)
    p.add_argument(
        "--cls-vote",
        type=str,
        default="attn_class",
        choices=["sum", "max", "topk", "lse", "attn_class"],
    )
    p.add_argument("--cls-lse-temp", type=float, default=1.0)
    p.add_argument("--cls-attn-hidden", type=int, default=128)
    p.add_argument("--cls-attn-dropout", type=float, default=0.1)
    p.add_argument("--cls-vote-clip", type=float, default=4.0)
    p.add_argument("--cls-vote-lam", type=float, default=0.35)
    p.add_argument("--cls-mix-bias-max", type=float, default=0.3)
    p.add_argument("--cls-threshold", type=float, default=0.5)
    p.add_argument("--cls-thresholds", type=str, default=DEFAULT_CLS_THRESHOLDS)
    p.add_argument("--cls-batch-size", type=int, default=8)
    p.add_argument("--cls-num-workers", type=int, default=1)
    p.add_argument("--cls-device", type=str, default=None)
    p.add_argument(
        "--cls-prob-plot-dir",
        type=Path,
        default=None,
        help="If set, save per-class probability distribution plots to this directory.",
    )
    p.add_argument("--cls-prob-plot-bins", type=int, default=50, help="Histogram bins for prob distribution plots.")
    return p.parse_args()


def ensure_dir(path: Path, clean: bool = False):
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_h5_case_ids(images_dir: Path) -> list[str]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"{images_dir} is not a directory")
    files = sorted(images_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found under {images_dir}")
    return [f.stem for f in files]


def load_case_ids(json_path: Optional[Path], images_dir: Path) -> list[str]:
    if json_path is None:
        return list_h5_case_ids(images_dir)
    with json_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError("--json must be a list")
    ids: List[str] = []
    for e in entries:
        if isinstance(e, dict):
            cid = e.get("case_id") or e.get("id") or e.get("name")
            if cid is None:
                raise ValueError(f"Split entry missing case identifier: {e}")
            ids.append(str(cid))
        else:
            ids.append(str(e))
    return ids


def write_case_json(case_ids: list[str], json_path: Path):
    data = [{"case_id": cid} for cid in case_ids]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _pick_preview_ids(case_ids: list[str], n: int, seed: int) -> list[str]:
    if n <= 0:
        return []
    if n >= len(case_ids):
        return list(case_ids)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(case_ids), size=n, replace=False)
    return [case_ids[i] for i in idx]


def _save_mask_overlay(img: np.ndarray, mask: np.ndarray, out_path: Path, alpha: float = 0.4) -> None:
    if plt is None:
        logger.warning("matplotlib not available; skipping visualization %s", out_path)
        return
    img_disp = img.astype(np.float32)
    if img_disp.max() > 1.5:
        img_disp = img_disp / 255.0
    img_disp = np.clip(img_disp, 0.0, 1.0)

    mask_int = mask.astype(np.int32)
    colors = np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 128, 0],
            [128, 0, 255],
            [0, 128, 255],
            [255, 0, 128],
            [0, 255, 128],
            [128, 255, 0],
            [128, 128, 255],
            [255, 128, 128],
            [128, 255, 128],
        ],
        dtype=np.float32,
    )
    color_map = colors[mask_int % len(colors)] / 255.0
    overlay = img_disp * (1.0 - alpha) + color_map * alpha

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=150)
    plt.close()


def to_tensor(img: np.ndarray) -> torch.Tensor:
    if img.ndim == 2:
        img = img[:, :, None]
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return torch.from_numpy(img).permute(2, 0, 1)


def _parse_thresholds(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    s = str(raw).strip()
    # tolerate quoted list strings like "[0.5,0.5,...]"
    if (s.startswith("\"") and s.endswith("\"")) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    if not s:
        return None
    if s.startswith("[") or s.startswith("{"):
        obj = json.loads(s)
        if isinstance(obj, dict) and "thresholds" in obj:
            obj = obj["thresholds"]
        if isinstance(obj, list):
            return [float(x) for x in obj]
        raise ValueError("Invalid thresholds JSON (expected list or {'thresholds': [...]})")
    return [float(x) for x in s.split(",") if x.strip()]


def load_seg_config(ckpt_path: Path) -> SegConfig:
    ckpt = torch_load_ckpt(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    img_size = int(args.get("img_size", 256))
    patch_size = args.get("patch_size", None)
    if isinstance(patch_size, (list, tuple)) and patch_size:
        patch_size = (int(patch_size[0]), int(patch_size[1] if len(patch_size) > 1 else patch_size[0]))
    elif isinstance(patch_size, str) and patch_size:
        parts = [int(x) for x in patch_size.replace("x", ",").split(",") if x.strip()]
        patch_size = (parts[0], parts[1]) if len(parts) == 2 else (parts[0], parts[0])
    elif patch_size is None:
        patch_size = None

    normalize = str(args.get("normalize", "per_image"))
    mean = tuple(float(x) for x in str(args.get("mean", "0.5,0.5,0.5")).split(",")) if normalize == "mean_std" else None
    std = tuple(float(x) for x in str(args.get("std", "0.5,0.5,0.5")).split(",")) if normalize == "mean_std" else None

    num_classes = int(args.get("num_classes", 2))
    in_channels = int(args.get("in_channels", 3))
    base_channels = int(args.get("base_channels", 32))
    bilinear = bool(args.get("bilinear", False))
    nnunet_plan = args.get("nnunet_plan", None)
    nnunet_plan = Path(nnunet_plan) if nnunet_plan else None
    if nnunet_plan is None or not nnunet_plan.exists():
        local_plan = Path(__file__).resolve().parent / "nnUNetPlans.json"
        if local_plan.exists():
            nnunet_plan = local_plan
    nnunet_plan_config = str(args.get("nnunet_plan_config", "2d"))
    plan_override = bool(args.get("plan_override", False))
    use_plan_arch = bool(args.get("use_plan_arch", False))
    if nnunet_plan is not None and nnunet_plan.exists():
        # Force PlanUNet when plan file is available to match trained checkpoints.
        use_plan_arch = True

    # Apply plan override (same as train_unet_chd.py)
    if nnunet_plan is not None and nnunet_plan.exists() and plan_override:
        plan_cfg = load_nnunet_plan(nnunet_plan, nnunet_plan_config)
        patch_size = tuple(plan_cfg["patch_size"])
        plan_norm = plan_cfg.get("normalization_schemes", [])
        if any("ZScore" in str(n) for n in plan_norm):
            normalize = "per_image"
            mean = None
            std = None

    return SegConfig(
        img_size=img_size,
        patch_size=patch_size,
        normalize=normalize,
        mean=mean,
        std=std,
        num_classes=num_classes,
        in_channels=in_channels,
        base_channels=base_channels,
        bilinear=bilinear,
        nnunet_plan=nnunet_plan,
        nnunet_plan_config=nnunet_plan_config,
        plan_override=plan_override,
        use_plan_arch=use_plan_arch,
    )


def load_nnunet_plan(plan_path: Path, config: str) -> Dict[str, object]:
    obj = json.loads(plan_path.read_text())
    if "configurations" not in obj:
        raise ValueError(f"Invalid plan file: {plan_path}")
    if config not in obj["configurations"]:
        raise ValueError(f"Config '{config}' not found in plan {plan_path}")
    return obj["configurations"][config]


def build_unet_model(cfg: SegConfig) -> torch.nn.Module:
    # Import UNet/PlanUNet from local train_unet_chd.py
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from train_unet_chd import PlanUNet, UNet  # type: ignore

    if cfg.use_plan_arch and cfg.nnunet_plan is not None and cfg.nnunet_plan.exists():
        plan_cfg = load_nnunet_plan(cfg.nnunet_plan, cfg.nnunet_plan_config)
        arch_kwargs = plan_cfg.get("architecture", {}).get("arch_kwargs")
        if not arch_kwargs:
            raise ValueError(f"Missing arch_kwargs in plan: {cfg.nnunet_plan}")
        return PlanUNet(cfg.in_channels, cfg.num_classes, arch_kwargs)
    return UNet(cfg.in_channels, cfg.num_classes, cfg.base_channels, cfg.bilinear)


def preprocess_image(img: np.ndarray, cfg: SegConfig) -> Tuple[torch.Tensor, Dict[str, int]]:
    img_t = to_tensor(img)
    _, h, w = img_t.shape
    meta: Dict[str, int] = {"orig_h": int(h), "orig_w": int(w), "mode": 0}

    if cfg.patch_size is not None:
        target_h, target_w = cfg.patch_size
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        if pad_h > 0 or pad_w > 0:
            img_t = F.pad(img_t, (0, pad_w, 0, pad_h), value=0.0)
        _, h2, w2 = img_t.shape
        top = 0
        left = 0
        if h2 != target_h or w2 != target_w:
            top = max(0, (h2 - target_h) // 2)
            left = max(0, (w2 - target_w) // 2)
            img_t = img_t[:, top : top + target_h, left : left + target_w]
        meta.update(
            {
                "mode": 1,
                "pad_h": int(pad_h),
                "pad_w": int(pad_w),
                "top": int(top),
                "left": int(left),
                "target_h": int(target_h),
                "target_w": int(target_w),
            }
        )
    elif cfg.img_size > 0 and (h != cfg.img_size or w != cfg.img_size):
        img_t = F.interpolate(
            img_t.unsqueeze(0),
            size=(cfg.img_size, cfg.img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        meta.update({"mode": 2, "target_h": int(cfg.img_size), "target_w": int(cfg.img_size)})

    if cfg.normalize == "per_image":
        mean = img_t.mean(dim=(1, 2), keepdim=True)
        std = img_t.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        img_t = (img_t - mean) / std
    elif cfg.normalize == "mean_std":
        if cfg.mean is None or cfg.std is None:
            raise ValueError("mean/std must be provided when normalize=mean_std")
        mean = torch.tensor(cfg.mean, dtype=img_t.dtype)[:, None, None]
        std = torch.tensor(cfg.std, dtype=img_t.dtype)[:, None, None]
        img_t = (img_t - mean) / std
    return img_t, meta


def restore_mask(pred: np.ndarray, meta: Dict[str, int]) -> np.ndarray:
    orig_h = int(meta["orig_h"])
    orig_w = int(meta["orig_w"])
    mode = int(meta.get("mode", 0))
    if mode == 1:
        target_h = int(meta["target_h"])
        target_w = int(meta["target_w"])
        top = int(meta["top"])
        left = int(meta["left"])
        pad_h = int(meta["pad_h"])
        pad_w = int(meta["pad_w"])
        if pad_h > 0 or pad_w > 0:
            return pred[:orig_h, :orig_w]
        out = np.zeros((orig_h, orig_w), dtype=pred.dtype)
        y1 = min(orig_h, top + target_h)
        x1 = min(orig_w, left + target_w)
        out[top:y1, left:x1] = pred[: y1 - top, : x1 - left]
        return out
    if mode == 2:
        if pred.shape[0] == orig_h and pred.shape[1] == orig_w:
            return pred
        t = torch.from_numpy(pred[None, None].astype(np.float32))
        t = F.interpolate(t, size=(orig_h, orig_w), mode="nearest")
        return t.squeeze(0).squeeze(0).to(torch.uint8).numpy()
    return pred


def run_segmentation(
    case_ids: list[str],
    images_dir: Path,
    label_dir: Path,
    ckpt_path: Path,
    device: torch.device,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    cfg = load_seg_config(ckpt_path)
    model = build_unet_model(cfg).to(device)
    ckpt = torch_load_ckpt(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    ensure_dir(label_dir, clean=True)
    masks: Dict[str, np.ndarray] = {}

    items: List[Tuple[str, torch.Tensor, Dict[str, int]]] = []
    for cid in case_ids:
        h5_path = images_dir / f"{cid}.h5"
        with h5py.File(h5_path, "r") as hf:
            if "image" not in hf:
                raise ValueError(f"Missing 'image' dataset in {h5_path}")
            img = hf["image"][()]
        img_t, meta = preprocess_image(img, cfg)
        items.append((cid, img_t, meta))

    if not items:
        return masks

    batch_size = max(1, int(batch_size))
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            imgs = torch.stack([x[1] for x in batch], dim=0).to(device, non_blocking=True)
            logits = model(imgs)
            if cfg.num_classes <= 1:
                pred = (torch.sigmoid(logits) > 0.5).long()
            else:
                pred = torch.argmax(logits, dim=1)
            pred_np = pred.detach().cpu().numpy().astype(np.uint8)
            for j, (cid, _img_t, meta) in enumerate(batch):
                mask = restore_mask(pred_np[j], meta)
                masks[cid] = mask
                out_path = label_dir / f"{cid}_label.h5"
                with h5py.File(out_path, "w") as hf:
                    hf.create_dataset("mask", data=mask, compression="gzip", dtype=np.uint8)
    return masks


def run_cls(args, json_path: Path, label_dir: Path, cls_out: Path):
    ensure_dir(cls_out, clean=True)
    if not args.cls_script.exists():
        raise FileNotFoundError(f"Classifier script not found: {args.cls_script}")
    json_path = json_path.resolve()
    label_dir = label_dir.resolve()
    cls_out = cls_out.resolve()
    images = args.images.resolve()
    use_ensemble = args.cls_script.name == "infer_p3_film_ensemble_3models.py"
    use_attnclass = args.cls_script.name == "infer_fc_patch_fullimg_film_attnclass.py"
    use_full_v2 = args.cls_script.name in {
        "infer_full_pos7_hist_rel_v2.py",
        "infer_full_pos7_hist_rel_v2_revise.py",
        "infer_full_pos7_hist_rel_v2_revise_vote.py",
        "infer_full_pos7_hist_rel_v2_revise_vote_v2.py",
        "infer_full_pos7_hist_rel_v3_viewmask_balanced_f1fix_hardneg_multiscale.py",
        "infer_full_pos7_hist_rel_v4_aug_topk_hardneg.py",
    }
    use_v4_topk = args.cls_script.name == "infer_full_pos7_hist_rel_v4_aug_topk_hardneg.py"
    use_vote = args.cls_script.name == "infer_full_pos7_hist_rel_v2_revise_vote.py"
    use_vote_v2 = args.cls_script.name == "infer_full_pos7_hist_rel_v2_revise_vote_v2.py"
    if use_ensemble:
        cmd = [
            sys.executable,
            str(args.cls_script),
            "--json", str(json_path),
            "--images", str(images),
            "--labels", str(label_dir),
            "--weights", str(args.cls_weights),
            "--ckpt-e", str(args.cls_ckpt),
            "--out-dir", str(cls_out),
            "--threshold", str(args.cls_threshold),
            "--batch-size", str(args.cls_batch_size),
            "--num-workers", str(args.cls_num_workers),
        ]
    else:
        local_scale = float(args.cls_local_scale)
        if (not use_attnclass) and local_scale < 0:
            denom = max(1, int(args.cls_locals_total))
            local_scale = float(args.cls_local_total_weight) / float(denom)
        cmd = [
            sys.executable,
            str(args.cls_script),
            "--json", str(json_path),
            "--images", str(images),
            "--labels", str(label_dir),
            "--weights", str(args.cls_weights),
            "--ckpt", str(args.cls_ckpt),
            "--out-dir", str(cls_out),
            "--img-size", str(args.cls_img_size),
            "--patch-size", str(args.cls_patch_size),
            "--locals-total", str(args.cls_locals_total),
            "--locals-greedy", str(args.cls_locals_greedy),
            "--min-dist-ratio", str(args.cls_min_dist_ratio),
            "--min-overlap-ratio", str(args.cls_min_overlap_ratio),
            "--bbox-margin", str(args.cls_bbox_margin),
            "--erode-kernel", str(args.cls_erode_kernel),
            "--erode-iters", str(args.cls_erode_iters),
            "--film-hidden", str(args.cls_film_hidden),
            "--film-scale", str(args.cls_film_scale),
            "--film-scale-pos", str(args.cls_film_scale_pos),
            "--film-scale-mask", str(args.cls_film_scale_mask),
            "--mask-cond", str(args.cls_mask_cond),
            "--mask-cond-dropout", str(args.cls_mask_cond_dropout),
            "--mask-hist-ids", str(args.cls_mask_hist_ids),
            "--mask-hist-patch-size", str(args.cls_mask_hist_patch_size),
            "--local-scale", str(local_scale),
            "--global-scale", str(args.cls_global_scale),
            "--threshold", str(args.cls_threshold),
            "--batch-size", str(args.cls_batch_size),
            "--num-workers", str(args.cls_num_workers),
        ]
        if use_v4_topk:
            cmd += [
                "--local-total-weight", str(args.cls_local_total_weight),
                "--agg-mode", str(args.cls_agg_mode),
                "--topk", str(args.cls_topk),
            ]
        if use_vote:
            cmd += [
                "--local-total-weight", str(args.cls_local_total_weight),
                "--vote-clip", str(args.cls_vote_clip),
                "--vote-lam", str(args.cls_vote_lam),
                "--mix-bias-scale", str(1.5),
            ]
        if use_vote_v2:
            cmd += [
                "--local-total-weight", str(args.cls_local_total_weight),
                "--vote-clip", str(args.cls_vote_clip),
                "--vote-lam", str(args.cls_vote_lam),
                "--mix-bias-max", str(args.cls_mix_bias_max),
            ]
        if use_full_v2:
            cmd += ["--disease-json", str(args.cls_disease_json)]
        if use_attnclass:
            cmd += [
                "--disease-json", str(args.cls_disease_json),
                "--local-total-weight", str(args.cls_local_total_weight),
                "--vote", str(args.cls_vote),
                "--topk", str(args.cls_topk),
                "--lse-temp", str(args.cls_lse_temp),
                "--attn-hidden", str(args.cls_attn_hidden),
                "--attn-dropout", str(args.cls_attn_dropout),
            ]
    if args.cls_device:
        cmd += ["--device", str(args.cls_device)]
    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    repo_root = Path(__file__).resolve().parent
    dinov3_root = repo_root / "dinov3"
    extra = str(dinov3_root) if dinov3_root.exists() else ""
    base = str(repo_root)
    parts = [base]
    if extra:
        parts.append(extra)
    if py_path:
        parts.append(py_path)
    env["PYTHONPATH"] = ":".join(parts)
    logger.info("Running classifier: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=repo_root)


def collect_cls_outputs(
    case_ids: list[str], cls_out: Path, thresholds: list[float] | None
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    labels: dict[str, np.ndarray] = {}
    probs: dict[str, np.ndarray] = {}
    th = np.asarray(thresholds, dtype=np.float32) if thresholds else None
    for cid in case_ids:
        path = cls_out / f"{cid}.h5"
        if not path.exists():
            raise FileNotFoundError(f"Classifier prediction missing: {path}")
        with h5py.File(path, "r") as hf:
            prob = hf["prob"][()].astype(np.float32) if "prob" in hf else None
            if prob is not None:
                probs[cid] = prob
            if th is not None:
                if prob is None:
                    raise ValueError(f"No 'prob' dataset in {path} for thresholding")
                if prob.shape[-1] != th.shape[0]:
                    raise ValueError(f"{path} prob shape {prob.shape} != thresholds {th.shape}")
                labels[cid] = (prob >= th).astype(np.uint8)
            else:
                if "label" not in hf:
                    raise ValueError(f"No 'label' dataset in {path}")
                labels[cid] = hf["label"][()].astype(np.uint8)
    return labels, probs


def save_final_outputs(
    case_ids: list[str],
    masks: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    out_dir: Path,
    probs: dict[str, np.ndarray] | None = None,
):
    ensure_dir(out_dir)
    for cid in case_ids:
        mask = masks[cid].astype(np.uint8)
        label = labels[cid].astype(np.uint8)
        prob = probs.get(cid) if probs is not None else None
        out_path = out_dir / f"{cid}.h5"
        with h5py.File(out_path, "w") as hf:
            hf.create_dataset("mask", data=mask, compression="gzip", dtype=np.uint8)
            hf.create_dataset("label", data=label, compression="gzip", dtype=np.uint8)
            if prob is not None:
                hf.create_dataset("prob", data=prob, compression="gzip", dtype=np.float32)


def plot_cls_prob_distributions(
    probs: dict[str, np.ndarray],
    out_dir: Path,
    bins: int = 50,
):
    if plt is None:
        logger.warning("matplotlib not available; skip prob distribution plots")
        return
    if not probs:
        logger.warning("No prob predictions found; skip prob distribution plots")
        return
    keys = sorted(probs.keys())
    prob_stack = np.stack([probs[k] for k in keys], axis=0)  # (N, K)
    if prob_stack.ndim != 2:
        logger.warning("Unexpected prob shape %s; skip plots", prob_stack.shape)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    num_classes = prob_stack.shape[1]
    for k in range(num_classes):
        vals = prob_stack[:, k]
        plt.figure(figsize=(6, 4))
        plt.hist(vals, bins=int(bins), range=(0.0, 1.0), color="#4C78A8", alpha=0.85, edgecolor="black", linewidth=0.4)
        plt.title(f"Class {k} prob distribution (N={len(vals)})")
        plt.xlabel("Probability")
        plt.ylabel("Count")
        mean = float(np.mean(vals))
        median = float(np.median(vals))
        plt.axvline(mean, color="#F58518", linestyle="--", linewidth=1.2, label=f"mean={mean:.3f}")
        plt.axvline(median, color="#54A24B", linestyle=":", linewidth=1.2, label=f"median={median:.3f}")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        out_path = out_dir / f"cls{k}_prob_dist.png"
        plt.savefig(out_path, dpi=150)
        plt.close()


def run_preview(args, case_ids: list[str]) -> None:
    preview_ids = _pick_preview_ids(case_ids, args.preview_num, args.preview_seed)
    if not preview_ids:
        return
    logger.info("Running preview on %d cases", len(preview_ids))

    temp_root = args.tmp_dir / "preview"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    seg_labels = temp_root / "seg_labels"
    cls_cases_json = temp_root / "cls_case_list.json"
    cls_out = temp_root / "cls_preds"
    vis_dir = args.preview_out

    device = torch.device(args.seg_device) if args.seg_device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masks = run_segmentation(
        case_ids=preview_ids,
        images_dir=args.images,
        label_dir=seg_labels,
        ckpt_path=args.seg_ckpt,
        device=device,
        batch_size=args.seg_batch_size,
    )
    write_case_json(preview_ids, cls_cases_json)
    run_cls(args, cls_cases_json, seg_labels, cls_out)

    thresholds = _parse_thresholds(args.cls_thresholds)
    cls_labels_map, _cls_probs_map = collect_cls_outputs(preview_ids, cls_out, thresholds)

    for cid in preview_ids:
        h5_path = args.images / f"{cid}.h5"
        with h5py.File(h5_path, "r") as hf:
            img = hf["image"][()]
            view = int(hf["view"][0]) if "view" in hf else -1
        mask = masks[cid]
        _save_mask_overlay(img, mask, vis_dir / f"{cid}_overlay.png")
        logger.info("Preview case %s view=%s cls_label=%s", cid, view, cls_labels_map[cid].tolist())
    logger.info("Preview visualizations saved to %s", vis_dir)


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    args = parse_args()
    device = torch.device(args.seg_device) if args.seg_device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.images is None and args.val_images is not None:
        args.images = args.val_images
    if args.images is None:
        raise ValueError("Missing --images (or --val-images) directory.")

    case_ids = load_case_ids(args.json, args.images)
    logger.info("Found %d cases", len(case_ids))

    temp_root = args.tmp_dir
    if args.keep_temp:
        ensure_dir(temp_root)
    else:
        if temp_root.exists():
            shutil.rmtree(temp_root)
        ensure_dir(temp_root)

    if args.preview_num and args.preview_num > 0:
        run_preview(args, case_ids)

    seg_labels = temp_root / "seg_labels"
    cls_cases_json = temp_root / "cls_case_list.json"
    cls_out = temp_root / "cls_preds"

    masks = run_segmentation(
        case_ids=case_ids,
        images_dir=args.images,
        label_dir=seg_labels,
        ckpt_path=args.seg_ckpt,
        device=device,
        batch_size=args.seg_batch_size,
    )
    write_case_json(case_ids, cls_cases_json)
    run_cls(args, cls_cases_json, seg_labels, cls_out)

    thresholds = _parse_thresholds(args.cls_thresholds)
    cls_labels_map, cls_probs_map = collect_cls_outputs(case_ids, cls_out, thresholds)

    save_final_outputs(case_ids, masks, cls_labels_map, args.out_dir, cls_probs_map)
    if args.cls_prob_plot_dir is not None:
        plot_cls_prob_distributions(cls_probs_map, args.cls_prob_plot_dir, bins=args.cls_prob_plot_bins)
    logger.info("Combined predictions written to %s", args.out_dir)
    if not args.keep_temp:
        shutil.rmtree(temp_root)


if __name__ == "__main__":
    main()
"""
python /gpfs/work/aac/haoyuwu24/bsaeline/FETUS-Challenge-ISBI2026/run_val_inference_unet_chd_combined.py \
  --val-images /gpfs/work/aac/haoyuwu24/val/images \
  --out-dir /gpfs/work/aac/haoyuwu24/bsaeline/FETUS-Challenge-ISBI2026/preds_unet_chd_combined_vote \
  --seg-ckpt /gpfs/work/aac/haoyuwu24/SupContrast/unet_runs/plan_701_revise/best.pth \
  --cls-script /gpfs/work/aac/haoyuwu24/dinov3/infer_full_pos7_hist_rel_v2_revise_vote_v2.py \
  --cls-ckpt /gpfs/work/aac/haoyuwu24/dinov3/outputs/FULL_pos7_hist_rel_vote_v2/best_fc_stage1_k4.pt \
  --cls-weights /gpfs/work/aac/haoyuwu24/dinov3/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --cls-disease-json /gpfs/work/aac/haoyuwu24/cls_index_find/disease_cases.json \
  --cls-locals-total 16 --cls-locals-greedy 16 \
  --cls-min-overlap-ratio 0.10 --cls-min-dist-ratio 0.5 \
  --cls-bbox-margin 0.15 --cls-erode-kernel 5 --cls-erode-iters 1 \
  --cls-local-total-weight 0.8 --cls-local-scale -1 --cls-global-scale 0.2 \
  --cls-film-hidden 64 --cls-film-scale-pos 0.02 --cls-film-scale-mask 0.005 \
  --cls-mask-cond hist --cls-mask-cond-dropout 0.5 \
  --cls-mask-hist-ids 1,2,3,4,5,6,7,8,9,12,13 --cls-mask-hist-patch-size 64 \
  --cls-vote-clip 4.0 --cls-vote-lam 0.35 \
  --cls-thresholds "[0.5,0.5,0.5,0.5,0.5,0.5,0.5]"



Classifier training reference (FULL_pos7_hist_rel_vote_v2):
PYTHONPATH=. python /gpfs/work/aac/haoyuwu24/dinov3/train_fc_patch_fullimg_film_v2_localweight_rarepatch_revise_vote_v2.py \
  --weights /gpfs/work/aac/haoyuwu24/dinov3/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --images /gpfs/work/aac/haoyuwu24/images \
  --labels /gpfs/work/aac/haoyuwu24/labels \
  --disease-json /gpfs/work/aac/haoyuwu24/cls_index_find/disease_cases.json \
  --output-dir /gpfs/work/aac/haoyuwu24/dinov3/outputs/FULL_pos7_hist_rel_vote_v2 \
  --epochs 200 --batch-size 8 --num-workers 1 \
  --unfreeze-blocks 4 --backbone-lr 2e-5 --lr 3e-4 --layer-decay 0.75 --warmup-epochs 10 \
  --locals-total 8 --locals-greedy 8 --bbox-margin 0.15 --min-overlap-ratio 0.10 --min-dist-ratio 0.5 \
  --mask-cond hist --mask-hist-ids 1,2,3,4,5,6,7,8,9,12,13 --mask-hist-patch-size 64 --mask-cond-dropout 0.5 \
  --film-hidden 64 --film-scale-pos 0.02 --film-scale-mask 0.005 \
  --vote-clip 4.0 --vote-lam 0.35 \
  --balance-batch --rare-classes "1,2,3" --rare-every 4 \
  --mix-bias-max 0.3

"""



