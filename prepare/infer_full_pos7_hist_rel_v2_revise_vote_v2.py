#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-model inference for the vote+FiLM model trained by:
  train_fc_patch_fullimg_film_v2_localweight_rarepatch_revise_vote_v2.py

Key NEW behavior requested:
(1) View-gating (hard suppress disallowed classes per view) before sigmoid
    - prevents "impossible classes" from contributing FP at deployment.
(2) Supports per-class thresholds via --cls-thresholds (optional).
    - If not provided, uses scalar --threshold.

Outputs one H5 per case_id with datasets:
  prob (float32,7), label (uint8,7)

NOTE:
- The official evaluation script can still apply its own thresholds; this script just produces
  probabilities and an optional binarized label.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import train_fc_patch_fullimg_film_v2_localweight_rarepatch_revise_vote_v2 as m_full


def parse_args():
    p = argparse.ArgumentParser("Infer FULL_pos7_hist_rel_vote (single model) with view-gating")
    p.add_argument("--json", type=Path, required=True, help="JSON list of case_ids (or dicts with case_id).")
    p.add_argument("--images", type=Path, required=True, help="Directory with {case_id}.h5")
    p.add_argument("--labels", type=Path, required=True, help="Directory with {case_id}_label.h5")
    p.add_argument(
        "--weights",
        type=Path,
        default=Path("/gpfs/work/aac/haoyuwu24/dinov3/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
    )
    p.add_argument(
        "--disease-json",
        type=Path,
        default=Path("/gpfs/work/aac/haoyuwu24/cls_index_find/disease_cases.json"),
        help="Used to infer num_classes (same as training).",
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Checkpoint path produced by the training script (best_fc_*.pt).",
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for prediction h5 files")

    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--patch-size", type=int, default=224)
    p.add_argument("--locals-total", type=int, default=8)
    p.add_argument("--locals-greedy", type=int, default=8)
    p.add_argument("--min-dist-ratio", type=float, default=0.5)
    p.add_argument("--min-overlap-ratio", type=float, default=0.1)

    # ROI + erosion
    p.add_argument("--bbox-margin", type=float, default=0.15)
    p.add_argument("--erode-kernel", type=int, default=5)
    p.add_argument("--erode-iters", type=int, default=1)

    # FiLM conditioning (mask variant)
    p.add_argument("--film-hidden", type=int, default=64)
    p.add_argument("--film-scale", type=float, default=0.02, help="[compat] used if --film-scale-pos not set")
    p.add_argument("--film-scale-pos", type=float, default=0.02)
    p.add_argument("--film-scale-mask", type=float, default=0.005)
    p.add_argument("--mask-cond", type=str, default="hist", choices=["none", "hist"])
    p.add_argument("--mask-cond-dropout", type=float, default=0.5)
    p.add_argument("--mask-hist-ids", type=str, default="1,2,3,4,5,6,7,8,9,12,13")
    p.add_argument("--mask-hist-patch-size", type=int, default=64)

    # aggregation weights (used to init vote mixing)
    p.add_argument("--local-total-weight", type=float, default=0.8)
    p.add_argument("--local-scale", type=float, default=-1.0)
    p.add_argument("--global-scale", type=float, default=0.2)

    # vote params
    p.add_argument("--vote-clip", type=float, default=4.0)
    p.add_argument("--vote-lam", type=float, default=0.35)

    # NEW: bounded mix_bias max must match training
    p.add_argument("--mix-bias-max", type=float, default=0.5)

    # NEW: view-gating mapping
    p.add_argument("--cls-allowed", type=str, default=None,
                   help="JSON string or .json path of {view_id:[cls_ids...]}; default uses training DEFAULT_CLS_ALLOWED.")
    p.add_argument("--gate-logit", type=float, default=-20.0,
                   help="Logit value assigned to disallowed classes before sigmoid (more negative -> stronger suppression).")

    # thresholds
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--cls-thresholds", type=str, default=None,
                   help="Optional per-class thresholds. Accepts JSON list or comma list, e.g. \"[0.5,0.3,...]\" or \"0.5,0.3,...\"")

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _normalize_ids(raw) -> List[str]:
    ids: List[str] = []
    for e in raw:
        if isinstance(e, dict):
            cid = e.get("case_id") or e.get("id") or e.get("name")
            if cid is None:
                raise ValueError(f"Entry missing case_id: {e}")
            ids.append(str(cid))
        else:
            ids.append(str(e))
    return ids


def load_disease_cases(json_path: Path) -> Dict[str, List[str]]:
    with json_path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at root of {json_path}; got {type(data)}")
    cases: Dict[str, List[str]] = {}
    for disease, ids in data.items():
        if not isinstance(ids, list):
            raise ValueError(f"Expected list of IDs for {disease} in {json_path}")
        cases[disease] = [str(cid) for cid in ids]
    return cases


def build_pos_labels(disease_cases: Dict[str, List[str]]) -> Dict[str, List[int]]:
    id_to_classes: Dict[str, List[int]] = {}
    for cls_idx, ids in enumerate(disease_cases.values()):
        for cid in ids:
            id_to_classes.setdefault(str(cid), []).append(cls_idx)
    return id_to_classes


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


class IdWrapper(Dataset):
    def __init__(self, base: Dataset, ids: List[str]):
        self.base = base
        self.ids = ids

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        return self.ids[idx], *item


@dataclass
class ModelSpec:
    ckpt_path: Path
    img_size: int = 224
    patch_size: int = 224
    min_dist_ratio: float = 0.5
    min_overlap_ratio: float = 0.1
    locals_total: int = 8
    locals_greedy: int = 8
    bbox_margin: float = 0.15
    erode_kernel: int = 5
    erode_iters: int = 1
    film_hidden: int = 64
    film_scale_pos: float = 0.02
    film_scale_mask: float = 0.005
    mask_cond: str = "hist"
    mask_hist_ids: Optional[List[int]] = None
    mask_hist_patch_size: int = 64
    mask_cond_dropout: float = 0.5
    local_scale: float = -1.0
    global_scale: float = 0.2
    vote_clip: float = 4.0
    vote_lam: float = 0.35
    out_scale: float = 1.0
    wL_init: float = 0.8
    mix_bias_max: float = 0.5


def _load_ckpt(path: Path) -> Dict:
    ckpt = torch.load(path, map_location="cpu")
    return ckpt.get("state_dict", ckpt)


def build_model(spec: ModelSpec, num_classes: int, weights: Path, mask_dim: Optional[int]) -> torch.nn.Module:
    backbone = m_full.dinov3_vitb16(weights=str(weights), pretrained=True)
    model = m_full.DinoPatchFullimgFiLMClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pos_dim=m_full.PatchPosFullimgDataset.POS_DIM,
        mask_dim=mask_dim or 0,
        film_hidden=spec.film_hidden,
        film_scale_pos=spec.film_scale_pos,
        film_scale_mask=spec.film_scale_mask,
        mask_cond_dropout=spec.mask_cond_dropout,
        local_scale=spec.local_scale,
        global_scale=spec.global_scale,
        head_hidden=256,
        head_dropout=0.1,
        head_layernorm=True,
        vote_clip=spec.vote_clip,
        vote_lam=spec.vote_lam,
        out_scale=spec.out_scale,
        wL_init=spec.wL_init,
        mix_bias_max=spec.mix_bias_max,
    )
    return model


def build_dataset(
    spec: ModelSpec,
    ids: List[str],
    id_to_cls: Dict[str, List[int]],
    images_dir: Path,
    labels_dir: Path,
    transform,
    num_classes: int,
) -> Dataset:
    return m_full.PatchPosFullimgDataset(
        ids=ids,
        images_dir=images_dir,
        labels_dir=labels_dir,
        id_to_cls=id_to_cls,
        transform=transform,
        patch_size=spec.patch_size,
        min_dist_ratio=spec.min_dist_ratio,
        min_overlap_ratio=spec.min_overlap_ratio,
        locals_total=spec.locals_total,
        locals_greedy=spec.locals_greedy,
        bbox_margin=spec.bbox_margin,
        erode_kernel=spec.erode_kernel,
        erode_iters=spec.erode_iters,
        mask_cond=spec.mask_cond,
        mask_hist_ids=spec.mask_hist_ids or [],
        mask_hist_patch_size=spec.mask_hist_patch_size,
        num_classes=num_classes,
    )


def parse_cls_thresholds(raw: Optional[str], num_classes: int, fallback: float) -> np.ndarray:
    if raw is None or str(raw).strip() == "":
        return np.full((num_classes,), float(fallback), dtype=np.float32)
    s = str(raw).strip()
    try:
        if s.startswith("[") or s.startswith("{"):
            arr = json.loads(s)
        else:
            arr = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
        arr = list(arr)
        if len(arr) != num_classes:
            raise ValueError(f"--cls-thresholds length {len(arr)} != num_classes {num_classes}")
        return np.asarray(arr, dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Failed to parse --cls-thresholds={raw}: {e}") from e


def run_infer(
    spec: ModelSpec,
    ids: List[str],
    id_to_cls: Dict[str, List[int]],
    images_dir: Path,
    labels_dir: Path,
    weights: Path,
    num_classes: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    cls_allowed: Dict[int, List[int]],
    gate_logit: float,
) -> Dict[str, np.ndarray]:
    transform = m_full.build_transform(spec.img_size)
    ds = build_dataset(spec, ids, id_to_cls, images_dir, labels_dir, transform, num_classes)
    wrapped = IdWrapper(ds, ids)
    loader = DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
    )

    mask_dim = getattr(ds, "mask_dim", None)
    model = build_model(spec, num_classes=num_classes, weights=weights, mask_dim=mask_dim)
    state_dict = _load_ckpt(spec.ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[infer] WARN missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)
    model.eval()

    out: Dict[str, np.ndarray] = {}
    gate_logit = float(gate_logit)

    with torch.no_grad():
        for batch in loader:
            # Dataset returns: patches, pos, mhist, label, view_id
            # IdWrapper adds cid at front -> (cid, patches, pos, mhist, label, view_id)
            cids, patches, pos, mhist, _label, view_id = batch
            patches = patches.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            mhist = mhist.to(device, non_blocking=True) if model.mask_mlp is not None else None

            logits = model(patches, pos, mhist)  # (B,K)

            # ===== NEW: hard view-gating =====
            views = torch.as_tensor(view_id, dtype=torch.int64, device=device)
            allowed = m_full.build_allowed_mask_torch(views, cls_allowed, num_classes, device=device)  # (B,K)
            logits = logits * allowed + gate_logit * (1.0 - allowed)

            prob = torch.sigmoid(logits).float().cpu().numpy()
            for i, cid in enumerate(cids):
                out[str(cid)] = prob[i]
    return out


def save_predictions(preds: Dict[str, np.ndarray], out_dir: Path, thresholds: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    thr = thresholds.astype(np.float32).reshape(1, -1)
    for cid, prob in preds.items():
        pred = (prob.reshape(1, -1) >= thr).astype(np.uint8).reshape(-1)
        out_path = out_dir / f"{cid}.h5"
        with h5py.File(out_path, "w") as f:
            f.create_dataset("prob", data=prob.astype(np.float32))
            f.create_dataset("label", data=pred.astype(np.uint8))


def main():
    args = parse_args()
    seed_all(args.seed)

    for pth in [args.json, args.images, args.labels, args.weights, args.disease_json, args.ckpt]:
        if not Path(pth).exists():
            raise FileNotFoundError(f"Path not found: {pth}")

    with args.json.open("r") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError("--json must be a list")
    ids = _normalize_ids(entries)

    disease_cases = load_disease_cases(args.disease_json)
    num_classes = len(disease_cases)
    id_to_cls = build_pos_labels(disease_cases)

    thresholds = parse_cls_thresholds(args.cls_thresholds, num_classes=num_classes, fallback=float(args.threshold))

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Num cases: {len(ids)}, num_classes: {num_classes}")
    print(f"thresholds: {thresholds.tolist()}")

    film_scale_pos = float(args.film_scale_pos) if args.film_scale_pos is not None else float(args.film_scale)

    if args.local_scale is not None and float(args.local_scale) >= 0:
        local_scale = float(args.local_scale)
    else:
        local_scale = float(args.local_total_weight) / float(max(1, args.locals_total))
    global_scale = float(args.global_scale)
    local_eff = float(args.locals_total) * local_scale
    global_eff = global_scale
    out_scale = local_eff + global_eff
    wL_init = local_eff / (local_eff + global_eff + 1e-6)

    cls_allowed = m_full.load_cls_allowed(args.cls_allowed, m_full.DEFAULT_CLS_ALLOWED)
    print(f"cls_allowed (view_id=1..4): {cls_allowed}")
    print(f"gate_logit: {float(args.gate_logit)}")

    spec = ModelSpec(
        ckpt_path=args.ckpt,
        img_size=args.img_size,
        patch_size=args.patch_size,
        min_dist_ratio=args.min_dist_ratio,
        min_overlap_ratio=args.min_overlap_ratio,
        locals_total=args.locals_total,
        locals_greedy=args.locals_greedy,
        bbox_margin=args.bbox_margin,
        erode_kernel=args.erode_kernel,
        erode_iters=args.erode_iters,
        film_hidden=args.film_hidden,
        film_scale_pos=film_scale_pos,
        film_scale_mask=args.film_scale_mask if args.mask_cond == "hist" else 0.0,
        mask_cond=args.mask_cond,
        mask_hist_ids=[int(x) for x in str(args.mask_hist_ids).split(",") if str(x).strip()],
        mask_hist_patch_size=args.mask_hist_patch_size,
        mask_cond_dropout=args.mask_cond_dropout,
        local_scale=local_scale,
        global_scale=global_scale,
        vote_clip=float(args.vote_clip),
        vote_lam=float(args.vote_lam),
        out_scale=out_scale,
        wL_init=wL_init,
        mix_bias_max=float(args.mix_bias_max),
    )

    preds = run_infer(
        spec=spec,
        ids=ids,
        id_to_cls=id_to_cls,
        images_dir=args.images,
        labels_dir=args.labels,
        weights=args.weights,
        num_classes=num_classes,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cls_allowed=cls_allowed,
        gate_logit=float(args.gate_logit),
    )

    save_predictions(preds, args.out_dir, thresholds)
    print(f"Saved predictions to: {args.out_dir}")


if __name__ == "__main__":
    main()
"""


训练（单模型）
PYTHONPATH=. python train_fc_patch_fullimg_film_v2_localweight_rarepatch_revise_vote_v2.py \
  --weights /gpfs/work/aac/haoyuwu24/dinov3/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --images /gpfs/work/aac/haoyuwu24/images \
  --labels /gpfs/work/aac/haoyuwu24/labels \
  --disease-json /gpfs/work/aac/haoyuwu24/cls_index_find/disease_cases.json \
  --output-dir /gpfs/work/aac/haoyuwu24/dinov3/outputs/FULL_pos7_hist_rel_vote_v2 \
  --mask-cond hist --mask-cond-dropout 0.15 \
  --balance-batch --rare-classes 1,2,3 --rare-every 4 --steps-per-epoch 450 \
  --locals-total 8 --locals-greedy 8 \
  --vote-clip 4.0 --vote-lam 0.35 \
  --unfreeze-blocks 8 --backbone-lr 2e-5 --lr 3e-4 \
  --mix-bias-max 0.5 \
  --mix-bias-pen-lambda 0.02

infer（带 view-gating）
PYTHONPATH=. python infer_full_pos7_hist_rel_v2_revise_vote_v2.py \
  --json /gpfs/work/aac/haoyuwu24/dinov3/outputs/xxx/splits/test.json \
  --images /gpfs/work/aac/haoyuwu24/images \
  --labels /gpfs/work/aac/haoyuwu24/labels \
  --ckpt /gpfs/work/aac/haoyuwu24/dinov3/outputs/FULL_pos7_hist_rel_vote_v2/best_fc_stage1_k4.pt \
  --out-dir /gpfs/work/aac/haoyuwu24/dinov3/outputs/FULL_pos7_hist_rel_vote_v2/preds_test \
  --mask-cond hist --mask-cond-dropout 0.0 \
  --locals-total 8 --locals-greedy 8 \
  --vote-clip 4.0 --vote-lam 0.35 \
  --mix-bias-max 0.3 \
  --gate-logit -20 
"""
