#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINOv3 ViT-B/16 + FC head, PATCH + FULL-IMAGE (global) training with P3 FiLM conditioning.

This version includes the requested fixes:

(1) view_id is 1..4 (NOT 0..3):
    - VIEW_4CH=1, VIEW_LVOT=2, VIEW_RVOT(or other)=3, VIEW_3VT=4
    - DEFAULT_CLS_ALLOWED keys updated to 1..4
    - view-specific patch proposals updated accordingly

(2) View-masked LOSS (TRAIN-time):
    - Only compute loss for classes allowed in that view.
    - Prevents "impossible negative gradients" killing rare classes.

(3) BalancedBatchSamplerRare FIX (no epoch step explosion):
    - Add --steps-per-epoch (default: ceil(len(train_ds)/batch_size))
    - Sampler length and iteration uses steps_per_epoch

(4) F1 calculation FIX (EVAL-time):
    - Macro-F1 averages only over classes that have positives in the evaluated (view-masked) region: (tp+fn)>0
    - Avoid penalizing classes that never appear in val.

Threshold remains fixed at 0.5 as requested.

NOTE: "merge internal test into val" behavior is kept exactly.

Model defaults:
- --unfreeze-blocks 4
- --film-hidden 64
- Head: 768 -> 256 -> K with LayerNorm + Dropout 0.1 (no BatchNorm)

NEW (vote only; F1 untouched):
- Learnable per-class mixing weights between local and global (sigmoid to keep stable).
- Tanh-Clip Mean + Disagreement Penalty on locals:
    local_cons = mean(tanh_clip(local_logits)) - lam * std(tanh_clip(local_logits))

MODS in this revision (your requests):
(A) mix_bias bounded by tanh:
    bias = mix_bias_max * tanh(bias_raw)
    add args: --mix-bias-max (default 0.5, recommend 0.3 if FP high)

(B) Per-class rare forcing:
    build rare_cls -> indices, force: sample cls (inverse-freq weighted) then sample index from that cls.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from dinov3.hub.backbones import dinov3_vitb16


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SPLIT_NAMES = ("train", "val", "test")
NUM_CLASSES = 7

# =========================
# view_id is 1..4
# =========================
VIEW_4CH = 1
VIEW_LVOT = 2
VIEW_OTHER = 3  # RVOT or other
VIEW_3VT = 4

# Default mapping aligned with your eval_dino_cls view-masked logic
# (keys are view_id=1..4)
DEFAULT_CLS_ALLOWED: Dict[int, List[int]] = {
    VIEW_4CH:   [0, 1],       # VSD, AVS/Atresia
    VIEW_LVOT:  [0, 2, 3],    # VSD, Ao Hypoplasia, Ao Valve Stenosis
    VIEW_OTHER: [4, 5],       # DORV, Pulm Valve Stenosis (example)
    VIEW_3VT:   [2, 5, 6],    # Ao Hypoplasia, Pulm Valve Stenosis, Right Aortic Arch
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Patch+full-image CHD classification with FiLM conditioning (partially finetuned DINO backbone)"
    )
    p.add_argument("--images", type=Path, default=Path("/gpfs/work/aac/haoyuwu24/images"))
    p.add_argument("--labels", type=Path, default=Path("/gpfs/work/aac/haoyuwu24/labels"))
    p.add_argument(
        "--weights",
        type=Path,
        default=Path("/gpfs/work/aac/haoyuwu24/dinov3/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
    )
    p.add_argument(
        "--disease-json",
        type=Path,
        default=Path("/gpfs/work/aac/haoyuwu24/cls_index_find/disease_cases.json"),
        help="JSON map of disease names to case IDs for defining positive labels.",
    )
    p.add_argument("--test-json", type=Path, default=None, help="Optional JSON list; ids inside are held out from train/val")
    p.add_argument(
        "--merge-test-json-into-train",
        action="store_true",
        help="If set, ids from --test-json are appended to train split (and removed from val/test).",
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs/dino_patch_fullimg_film"))
    p.add_argument(
        "--split-json-dir",
        type=Path,
        default=None,
        help="Directory to dump train/val/test split JSONs (defaults to <output-dir>/splits)",
    )
    p.add_argument(
        "--load-split-json-dir",
        type=Path,
        default=None,
        help="Load existing train/val/test split JSONs instead of creating new ones.",
    )

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4, help="Head LR")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--patch-size", type=int, default=224)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    # local patch sampling
    p.add_argument("--min-dist-ratio", type=float, default=0.5, help="Greedy stage min distance as ratio of local_size.")
    p.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=0.1,
        help="Greedy stage: min mask overlap ratio (patch area). Recommended 0.05~0.15.",
    )
    p.add_argument("--locals-total", type=int, default=8, help="Number of LOCAL patches (global full-image patch will be appended).")
    p.add_argument("--locals-greedy", type=int, default=8, help="Greedy locals (<= locals-total).")

    # bbox margin + mask erosion controls
    p.add_argument(
        "--bbox-margin",
        type=float,
        default=0.15,
        help="ROI bbox expansion ratio computed on bbox size. Suggested 0.10~0.20.",
    )
    p.add_argument(
        "--erode-kernel",
        type=int,
        default=5,
        help="Binary erosion kernel size (odd). Used only for greedy center sampling.",
    )
    p.add_argument(
        "--erode-iters",
        type=int,
        default=1,
        help="Binary erosion iterations. Used only for greedy center sampling.",
    )

    # aggregation weights (kept for compatibility/printing)
    p.add_argument("--local-total-weight", type=float, default=0.8, help="Total weight assigned to all locals combined.")
    p.add_argument(
        "--local-scale",
        type=float,
        default=-1.0,
        help="Scale for sum(local_logits). If <0, auto = local_total_weight/locals_total.",
    )
    p.add_argument("--global-scale", type=float, default=0.2, help="Scale for global_logit (full image).")

    # NEW: robust vote hyperparams (locals aggregation)
    p.add_argument("--vote-clip", type=float, default=4.0, help="tanh clip constant c for local logits.")
    p.add_argument("--vote-lam", type=float, default=0.35, help="disagreement penalty lambda for local std.")

    # MOD(A): bounded bias max
    p.add_argument(
        "--mix-bias-max",
        type=float,
        default=0.5,
        help="Per-class bias bound: bias = mix_bias_max * tanh(bias_raw). Set 0 to disable bias.",
    )

    # FiLM conditioning
    p.add_argument("--film-hidden", type=int, default=64, help="Hidden dim of FiLM MLP(s).")
    p.add_argument("--film-scale", type=float, default=0.02, help="[compat] FiLM scale for pos branch (if --film-scale-pos not set).")
    p.add_argument("--film-scale-pos", type=float, default=None, help="FiLM scale for position branch (overrides --film-scale).")
    p.add_argument("--film-scale-mask", type=float, default=0.005, help="FiLM scale for mask-hist branch (should be smaller).")

    # weak mask conditioning
    p.add_argument("--mask-cond", type=str, default="hist", choices=["none", "hist"], help="Use weak mask statistics as conditioning.")
    p.add_argument("--mask-hist-ids", type=str, default="1,2,3,4,5,6,7,8,9,12,13", help="Mask label ids for histogram (exclude 0).")
    p.add_argument("--mask-hist-patch-size", type=int, default=64, help="Downsample size for mask-hist stats per patch.")
    p.add_argument("--mask-cond-dropout", type=float, default=0.5, help="Drop prob for mask conditioning per patch during training.")

    # loss
    p.add_argument("--pos-weight-cap", type=float, default=10.0, help="Clamp pos_weight to this max.")
    p.add_argument("--asym-gamma-pos", type=float, default=0.0)
    p.add_argument("--asym-gamma-neg", type=float, default=4.0)
    p.add_argument("--asym-clip", type=float, default=0.05)
    p.add_argument("--asym-eps", type=float, default=1e-8)

    # batching
    p.add_argument("--balance-batch", action="store_true", help="Balanced batch: each batch ~1:1 pos/neg")
    p.add_argument("--rare-classes", type=str, default="1,2,3", help="Comma list of rare class indices to force appear.")
    p.add_argument("--rare-every", type=int, default=4, help="Force a rare-positive at least once every N batches (>=1).")

    # FIX: prevent epoch length explosion
    p.add_argument(
        "--steps-per-epoch",
        type=int,
        default=-1,
        help="(balance-batch only) number of batches per epoch. If <=0, auto = ceil(len(train_ds)/batch_size).",
    )

    p.add_argument(
        "--cls-allowed",
        type=str,
        default=None,
        help="JSON string or .json path of {view_id:[cls_ids...]}; default uses DEFAULT_CLS_ALLOWED (view_id=1..4)",
    )

    # finetune
    p.add_argument("--unfreeze-blocks", type=int, default=4, help="How many of the final transformer blocks to unfreeze.")
    p.add_argument("--backbone-lr", type=float, default=2e-5, help="Base LR for unfrozen backbone blocks (before layer-wise decay).")
    p.add_argument("--layer-decay", type=float, default=0.75, help="LR decay between adjacent unfrozen blocks.")
    p.add_argument("--warmup-epochs", type=int, default=10, help="Linear warmup epochs before cosine.")
    p.add_argument("--grad-clip", type=float, default=3.0, help="Clip grad norm (<=0 disables).")
    p.add_argument("--init-checkpoint", type=Path, default=None, help="Optional checkpoint (state_dict) to load before training.")
    p.add_argument("--early-stop-patience", type=int, default=0, help="Early stop if val_f1 doesn't improve for N epochs (0 disables).")

    # staged unfreeze (optional)
    p.add_argument("--stage-unfreeze", type=str, default="", help="Comma-separated unfreeze blocks per stage (e.g., '2,4,8').")
    p.add_argument("--stage-epochs", type=str, default="", help="Comma-separated epochs per stage.")
    p.add_argument("--stage-head-lrs", type=str, default="", help="Comma-separated head LR per stage.")
    p.add_argument("--stage-backbone-lrs", type=str, default="", help="Comma-separated backbone LR per stage.")

    # nnUNet conversion options (kept; safe to ignore)
    p.add_argument("--nnunet-task-id", type=str, default="003")
    p.add_argument("--nnunet-dataset-name", type=str, default="Dataset003_PreTraining")
    p.add_argument("--nnunet-raw-root", type=Path, default=Path("/gpfs/work/aac/haoyuwu24/nnUNet_raw"))
    p.add_argument("--nnunet-preprocessed-root", type=Path, default=Path("/gpfs/work/aac/haoyuwu24/nnUNet_preprocessed"))
    p.add_argument("--nnunet-plan", action="store_true")
    p.add_argument("--nnunet-verify", action="store_true")
    p.add_argument("--convert-nnunet-to-h5", action="store_true")
    p.add_argument("--nnunet-h5-root", type=Path, default=Path("/gpfs/work/aac/haoyuwu24/dinov3/FAKE_LABEL_pretraining"))
    p.add_argument("--nnunet-convert-classes", type=int, default=NUM_CLASSES)

    return p.parse_args()


def _load_json_arg(s: Optional[str]):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    if s.startswith("{") or s.startswith("["):
        return json.loads(s)
    with open(s, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cls_allowed(arg: Optional[str], default: Dict[int, List[int]]) -> Dict[int, List[int]]:
    raw = _load_json_arg(arg)
    if raw is None:
        return default
    if not isinstance(raw, dict):
        raise ValueError("--cls-allowed must be a JSON object: {view_id: [cls_ids...]}")
    out: Dict[int, List[int]] = {}
    for k, v in raw.items():
        kk = int(k)
        if not isinstance(v, (list, tuple)):
            raise ValueError(f"cls_allowed[{k}] must be a list.")
        out[kk] = [int(x) for x in v]
    return out


def build_allowed_mask_torch(
    views: torch.Tensor,                 # (B,) int64 view_id in 1..4
    cls_allowed: Dict[int, List[int]],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    B = int(views.shape[0])
    K = int(num_classes)
    mask = torch.zeros((B, K), dtype=torch.float32, device=device)
    for i, v in enumerate(views.tolist()):
        for k in cls_allowed.get(int(v), []):
            if 0 <= k < K:
                mask[i, k] = 1.0
    return mask


def _parse_csv_list(raw: str, cast_fn):
    s = str(raw).strip()
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(cast_fn(part))
    return out


def _get_stage_value(values: List, idx: int, default):
    if not values:
        return default
    if idx < len(values):
        return values[idx]
    return values[-1]


def run_nnunet_plan_and_preprocess(task_id: str, raw_root: Path, preprocessed_root: Path, verify: bool) -> None:
    env = os.environ.copy()
    env["nnUNet_raw_data_base"] = str(raw_root)
    env["nnUNet_preprocessed"] = str(preprocessed_root)
    cmd = ["nnUNet_plan_and_preprocess", "-t", task_id]
    if verify:
        cmd.append("--verify_dataset_integrity")
    print(f"Running nnUnet plan+preprocess: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, env=env)
    except FileNotFoundError as exc:
        raise RuntimeError("`nnUNet_plan_and_preprocess` not found; install nnUNet or adjust PATH.") from exc


def convert_pretraining_to_h5(dataset_root: Path, output_root: Path, script_path: Path, classes: int) -> None:
    if not dataset_root.exists():
        raise FileNotFoundError(f"nnUNet dataset root not found: {dataset_root}")
    if not script_path.exists():
        raise FileNotFoundError(f"Conversion script missing: {script_path}")
    cmd = [
        sys.executable,
        str(script_path),
        "--nnunet-root",
        str(dataset_root),
        "--output-root",
        str(output_root),
        "--classes",
        str(classes),
    ]
    print(f"Running nnUnet-to-h5 conversion: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transform(img_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def collect_all_ids(images_dir: Path, labels_dir: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for h5_path in sorted(images_dir.glob("*.h5")):
        cid = h5_path.stem
        lbl_path = labels_dir / f"{cid}_label.h5"
        if not lbl_path.exists():
            continue
        with h5py.File(h5_path, "r") as f:
            view = int(f["view"][0]) if "view" in f else VIEW_4CH
        out[cid] = view
    return out


def load_disease_cases(json_path: Path) -> Dict[str, List[str]]:
    with json_path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict at root of {json_path}; got {type(data)}")
    cases: Dict[str, List[str]] = {}
    for disease, ids in data.items():
        if not isinstance(ids, list):
            raise ValueError(f"Expected list of IDs for {disease} in {json_path}")
        normalized = [str(cid) for cid in ids]
        cases[disease] = list(dict.fromkeys(normalized))
    return cases


def build_pos_labels(disease_cases: Dict[str, List[str]]) -> Tuple[List[str], Dict[str, List[int]]]:
    id_to_classes: Dict[str, List[int]] = {}
    for cls_idx, ids in enumerate(disease_cases.values()):
        for cid in ids:
            id_to_classes.setdefault(str(cid), []).append(cls_idx)
    pos_ids = list(id_to_classes.keys())
    return pos_ids, id_to_classes


def load_ids_from_json(json_path: Path) -> List[str]:
    with json_path.open("r") as f:
        data = json.load(f)
    ids: List[str] = []
    for e in data:
        if isinstance(e, dict) and "case_id" in e:
            ids.append(str(e["case_id"]))
        else:
            ids.append(str(e))
    return ids


def compute_class_members(all_ids: Dict[str, int], disease_cases: Dict[str, List[str]]) -> Dict[int, List[str]]:
    class_members: Dict[int, List[str]] = {}
    for cls_idx, ids in enumerate(disease_cases.values()):
        filtered = [cid for cid in ids if cid in all_ids]
        class_members[cls_idx] = list(dict.fromkeys(filtered))
    return class_members


def stratified_split_positive_ids(
    id_to_cls: Dict[str, List[int]],
    class_members: Dict[int, List[str]],
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    class_targets: Dict[int, Dict[str, int]] = {}
    for cls_idx, members in class_members.items():
        total = len(members)
        base, rem = divmod(total, 3)
        class_targets[cls_idx] = {"train": base + rem, "val": base, "test": base}

    assigned: Dict[int, Dict[str, int]] = {cls_idx: {split: 0 for split in SPLIT_NAMES} for cls_idx in class_targets}
    rng = random.Random(seed)
    ids = list(id_to_cls.keys())
    rng.shuffle(ids)
    assignment: Dict[str, str] = {}

    def _can_assign(cls_idx: int, split: str) -> bool:
        return assigned[cls_idx][split] < class_targets[cls_idx][split]

    def _shortage(classes: List[int], split: str) -> int:
        return sum(max(class_targets[c][split] - assigned[c][split], 0) for c in classes)

    for cid in ids:
        classes = id_to_cls[cid]
        split_choice = None
        for split in ("val", "test"):
            if all(_can_assign(c, split) for c in classes) and _shortage(classes, split) > 0:
                split_choice = split
                break
        if split_choice is None:
            for split in SPLIT_NAMES:
                if all(_can_assign(c, split) for c in classes):
                    split_choice = split
                    break
        if split_choice is None:
            split_choice = "train"

        assignment[cid] = split_choice
        for c in classes:
            assigned[c][split_choice] += 1

    train_ids = [cid for cid, split in assignment.items() if split == "train"]
    val_ids = [cid for cid, split in assignment.items() if split == "val"]
    test_ids = [cid for cid, split in assignment.items() if split == "test"]
    return train_ids, val_ids, test_ids


def split_negatives_evenly(neg_ids: Sequence[str], seed: int) -> Tuple[List[str], List[str], List[str]]:
    rng = random.Random(seed)
    neg_list = list(neg_ids)
    rng.shuffle(neg_list)
    split_point = math.ceil(len(neg_list) / 2)
    train_negs = neg_list[:split_point]
    test_negs = neg_list[split_point:]
    val_neg_count = min(len(train_negs), int(round(len(train_negs) * 0.2)))
    val_negs: List[str] = rng.sample(train_negs, val_neg_count) if val_neg_count > 0 else []
    val_set = set(val_negs)
    train_negs_final = [cid for cid in train_negs if cid not in val_set]
    return train_negs_final, val_negs, test_negs


def write_split_json(path: Path, ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(sorted(ids), f, indent=2)


# =========================
# pos_weight computed under view-masked counting (important)
# =========================
def compute_pos_weight_viewmasked(
    train_ids: Sequence[str],
    labels_all: Dict[str, List[int]],
    all_views: Dict[str, int],
    cls_allowed: Dict[int, List[int]],
    num_classes: int,
    cap: float,
) -> torch.Tensor:
    pos = np.zeros(num_classes, dtype=np.float32)
    tot = np.zeros(num_classes, dtype=np.float32)

    for cid in train_ids:
        v = int(all_views.get(cid, VIEW_4CH))
        allowed = cls_allowed.get(v, [])
        if not allowed:
            continue
        tot[np.array(allowed, dtype=np.int64)] += 1.0

        cls_list = labels_all.get(cid, [])
        for c in cls_list:
            if c in allowed:
                pos[int(c)] += 1.0

    neg = tot - pos
    ratio = np.ones(num_classes, dtype=np.float32)
    valid = (pos > 0) & (tot > 0)
    ratio[valid] = neg[valid] / pos[valid]
    pos_weight = np.sqrt(ratio)
    pos_weight = np.clip(pos_weight, 1.0, cap)
    return torch.tensor(pos_weight, dtype=torch.float32)


class AsymmetricLossMultiLabel(nn.Module):
    """
    Supports reduction='mean' or 'none' (elementwise BxK).
    """
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        pos_weight: torch.Tensor = None,
    ):
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip) if clip is not None else 0.0
        self.eps = float(eps)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.clone().detach())
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """
        logits: (B,K)
        targets: (B,K)
        return:
          - reduction='none': (B,K)
          - reduction='mean': scalar
        """
        x_prob = torch.sigmoid(logits)

        if self.clip is not None and self.clip > 0:
            x_neg = (1.0 - x_prob).clamp(min=self.clip, max=1.0)
        else:
            x_neg = 1.0 - x_prob

        x_pos = x_prob
        t_pos = targets
        t_neg = 1.0 - targets

        log_pos = torch.clamp(x_pos, self.eps, 1.0 - self.eps).log()
        log_neg = torch.clamp(x_neg, self.eps, 1.0 - self.eps).log()
        loss_pos = -t_pos * log_pos
        loss_neg = -t_neg * log_neg

        if self.gamma_pos > 0:
            loss_pos = loss_pos * torch.pow(1.0 - x_pos, self.gamma_pos)
        if self.gamma_neg > 0:
            loss_neg = loss_neg * torch.pow(x_prob, self.gamma_neg)

        loss = loss_pos + loss_neg

        if self.pos_weight is not None:
            loss = loss * (1.0 + (self.pos_weight - 1.0) * t_pos)

        if reduction == "none":
            return loss
        if reduction == "mean":
            return loss.mean()
        raise ValueError(f"Unknown reduction={reduction}")


def mask_bbox(mask: np.ndarray, margin: float = 0.0) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    if len(xs) == 0 or len(ys) == 0:
        return w // 4, h // 4, 3 * w // 4, 3 * h // 4
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    x1 = max(0, int(x1 - margin * bw))
    x2 = min(w - 1, int(x2 + margin * bw))
    y1 = max(0, int(y1 - margin * bh))
    y2 = min(h - 1, int(y2 + margin * bh))
    return x1, y1, x2, y2


def binary_erode(mask01: np.ndarray, k: int = 5, iters: int = 1) -> np.ndarray:
    """Binary erosion on {0,1} mask."""
    m = (mask01 > 0).astype(np.uint8)
    k = int(k)
    if k <= 1:
        return m
    if k % 2 == 0:
        k += 1
    iters = max(int(iters), 1)

    try:
        import cv2  # type: ignore
        kernel = np.ones((k, k), np.uint8)
        out = m.copy()
        for _ in range(iters):
            out = cv2.erode(out, kernel, iterations=1)
        return (out > 0).astype(np.uint8)
    except Exception:
        pass

    try:
        from numpy.lib.stride_tricks import sliding_window_view  # type: ignore
        out = m.copy()
        pad = k // 2
        for _ in range(iters):
            padded = np.pad(out, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
            win = sliding_window_view(padded, (k, k))
            out = win.min(axis=(-1, -2)).astype(np.uint8)
        return out
    except Exception as exc:
        raise RuntimeError("binary_erode needs either OpenCV (cv2) or numpy>=1.20 (sliding_window_view).") from exc


def _crop_and_resize_mask(mask: np.ndarray, sx: int, sy: int, ex: int, ey: int, out_size: int) -> np.ndarray:
    """Crop then resize to (out_size,out_size) via nearest."""
    h, w = mask.shape

    pad_left = max(0, -sx)
    pad_top = max(0, -sy)
    pad_right = max(0, ex - w)
    pad_bottom = max(0, ey - h)

    sx0 = max(0, sx)
    sy0 = max(0, sy)
    ex0 = min(w, ex)
    ey0 = min(h, ey)

    crop = mask[sy0:ey0, sx0:ex0]
    if pad_left or pad_right or pad_top or pad_bottom:
        crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)

    target_h = max(1, ey - sy)
    target_w = max(1, ex - sx)
    if crop.shape[0] != target_h or crop.shape[1] != target_w:
        pad_h = target_h - crop.shape[0]
        pad_w = target_w - crop.shape[1]
        if pad_h > 0 or pad_w > 0:
            crop = np.pad(crop, ((0, max(0, pad_h)), (0, max(0, pad_w))), mode="constant", constant_values=0)
        crop = crop[:target_h, :target_w]

    if crop.shape[0] != out_size or crop.shape[1] != out_size:
        try:
            import cv2  # type: ignore
            crop = cv2.resize(crop.astype(np.int32), (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        except Exception:
            if crop.shape[0] < out_size or crop.shape[1] < out_size:
                crop = np.pad(
                    crop,
                    ((0, max(0, out_size - crop.shape[0])), (0, max(0, out_size - crop.shape[1]))),
                    mode="constant",
                    constant_values=0,
                )
            crop = crop[:out_size, :out_size]
    return crop.astype(np.int32)


def compute_mask_hist_ratio(mask_crop: np.ndarray, ids: List[int]) -> np.ndarray:
    """returns m = [fg_ratio, ratio(id1), ratio(id2), ...] float32"""
    area = float(mask_crop.size) if mask_crop.size > 0 else 1.0
    fg_ratio = float((mask_crop > 0).sum()) / area
    ratios = [fg_ratio]
    for k in ids:
        ratios.append(float((mask_crop == int(k)).sum()) / area)
    return np.asarray(ratios, dtype=np.float32)


def _boundary_pixels(mask01: np.ndarray) -> np.ndarray:
    """Return boundary pixels (binary) via 4-neighborhood."""
    m = (mask01 > 0).astype(np.uint8)
    if m.sum() == 0:
        return m
    up = np.pad(m[1:, :], ((0, 1), (0, 0)), mode="constant")
    down = np.pad(m[:-1, :], ((1, 0), (0, 0)), mode="constant")
    left = np.pad(m[:, 1:], ((0, 0), (0, 1)), mode="constant")
    right = np.pad(m[:, :-1], ((0, 0), (1, 0)), mode="constant")
    eroded = m & up & down & left & right
    return (m & (1 - eroded)).astype(np.uint8)


def _adjacency_points(mask: np.ndarray, a: int, b: int) -> np.ndarray:
    """
    Pixels where label==a touches label==b (4-neighborhood). Return coordinates (y,x) on 'a' side.
    """
    A = (mask == int(a)).astype(np.uint8)
    if A.sum() == 0:
        return np.zeros((0, 2), dtype=np.int32)
    B = (mask == int(b)).astype(np.uint8)
    if B.sum() == 0:
        return np.zeros((0, 2), dtype=np.int32)

    Bu = np.pad(B[1:, :], ((0, 1), (0, 0)), mode="constant")
    Bd = np.pad(B[:-1, :], ((1, 0), (0, 0)), mode="constant")
    Bl = np.pad(B[:, 1:], ((0, 0), (0, 1)), mode="constant")
    Br = np.pad(B[:, :-1], ((0, 0), (1, 0)), mode="constant")
    touch = A & (Bu | Bd | Bl | Br)
    pts = np.argwhere(touch > 0).astype(np.int32)
    return pts


def _centroid(mask: np.ndarray, label: int) -> Optional[Tuple[float, float]]:
    pts = np.argwhere(mask == int(label))
    if pts.size == 0:
        return None
    y = float(pts[:, 0].mean())
    x = float(pts[:, 1].mean())
    return (y, x)


def _bbox(mask: np.ndarray, label: int) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask == int(label))
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2


def _safe_ratio(a: float, b: float, eps: float = 1e-6) -> float:
    return float(a) / float(b + eps)


def compute_extra_mask_features(mask: np.ndarray, view_id: int) -> np.ndarray:
    """
    Fixed-dim extra features appended to every patch mhist.
    dims (10):
      0 lv_rv_ratio
      1 la_lv_ratio
      2 ra_rv_ratio
      3 av_junction_offset (abs y_diff normalized) [4CH]
      4 aao_lv_ratio [LVOT]
      5 arch_mpa_ratio [3VT]
      6 lv_roundness (4*pi*area/perim^2) [LVOT]
      7 aao_diam_norm (equiv diameter / min(H,W)) [LVOT]
      8 aao_bbox_aspect (w/h) [LVOT]
      9 arch_diam_norm (equiv diameter / min(H,W)) [3VT]
    """
    h, w = mask.shape
    mn = float(min(h, w))
    feats = np.zeros((10,), dtype=np.float32)

    area_la = float((mask == 1).sum())
    area_lv = float((mask == 2).sum())
    area_ra = float((mask == 3).sum())
    area_rv = float((mask == 4).sum())
    area_aao = float((mask == 8).sum())
    area_mpa = float((mask == 9).sum())
    area_arch = float((mask == 13).sum())

    feats[0] = _safe_ratio(area_lv, area_rv)
    feats[1] = _safe_ratio(area_la, area_lv)
    feats[2] = _safe_ratio(area_ra, area_rv)

    if int(view_id) == VIEW_4CH:
        pts_l = _adjacency_points(mask, 1, 2)
        pts_r = _adjacency_points(mask, 3, 4)
        if pts_l.size > 0 and pts_r.size > 0:
            yl = float(pts_l[:, 0].mean())
            yr = float(pts_r[:, 0].mean())
            feats[3] = float(abs(yl - yr) / float(max(h, 1)))
        else:
            feats[3] = 0.0

    if int(view_id) == VIEW_LVOT:
        feats[4] = _safe_ratio(area_aao, area_lv)

        LV = (mask == 2).astype(np.uint8)
        area = float(LV.sum())
        if area > 0:
            perim = float(_boundary_pixels(LV).sum())
            if perim > 0:
                feats[6] = float(4.0 * math.pi * area / (perim * perim + 1e-6))

        if area_aao > 0:
            diam = math.sqrt(4.0 * area_aao / math.pi)
            feats[7] = float(diam / (mn + 1e-6))
            bb = _bbox(mask, 8)
            if bb is not None:
                x1, y1, x2, y2 = bb
                bw = float(max(1, x2 - x1 + 1))
                bh = float(max(1, y2 - y1 + 1))
                feats[8] = float(bw / bh)

    if int(view_id) == VIEW_3VT:
        if area_arch > 0 and area_mpa > 0:
            feats[5] = _safe_ratio(area_arch, area_mpa)
            diam_arch = math.sqrt(4.0 * area_arch / math.pi)
            feats[9] = float(diam_arch / (mn + 1e-6))

    return feats.astype(np.float32)


def extract_patches_with_pos_fullimg(
    img: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
    view_id: int = VIEW_4CH,
    min_dist_ratio: float = 0.5,
    min_overlap_ratio: float = 0.1,
    locals_total: int = 8,
    locals_greedy: int = 8,
    bbox_margin: float = 0.15,
    erode_kernel: int = 5,
    erode_iters: int = 1,
    mask_hist_ids: Optional[List[int]] = None,
    mask_hist_patch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Local patches (N=locals_total) + 1 global full-image patch, with:
    - pos vector p_i (dim=7): [cx_img, cy_img, cx_roi, cy_roi, s_rel, is_global, view_norm]
    - weak mask vector m_i: hist ratios + extra geometry scalars (fixed dim)

    Return:
      patches: (locals_total+1, patch_size, patch_size, 3)
      pos:     (locals_total+1, 7)
      mhist:   (locals_total+1, Dm) where Dm = (1+len(mask_hist_ids)) + EXTRA_DIM  (or 0 if None)
    """
    h, w, _ = img.shape
    x1, y1, x2, y2 = mask_bbox(mask, margin=float(bbox_margin))
    bw, bh = max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)

    S = max(bw, bh)
    base_local_size = int(np.clip(round(0.35 * S), 96, 192))
    min_dist = base_local_size * float(min_dist_ratio)

    locals_greedy = max(0, min(int(locals_greedy), int(locals_total)))
    locals_total = max(locals_greedy, int(locals_total))

    mask01 = (mask > 0).astype(np.uint8)

    proposals: List[Tuple[float, float, int]] = []
    extra = compute_extra_mask_features(mask, view_id=view_id)

    if int(view_id) == VIEW_4CH:
        junction_size = int(np.clip(round(0.22 * S), 96, 160))
        pts_l = _adjacency_points(mask, 1, 2)
        pts_r = _adjacency_points(mask, 3, 4)

        def _pick_center(pts: np.ndarray) -> Optional[Tuple[float, float]]:
            if pts.size == 0:
                return None
            i = int(len(pts) // 2)
            p = pts[i]
            return float(p[0]), float(p[1])

        c1 = _pick_center(pts_l)
        c2 = _pick_center(pts_r)
        if c1 is not None:
            proposals.append((c1[0], c1[1], junction_size))
        if c2 is not None:
            proposals.append((c2[0], c2[1], junction_size))

    if int(view_id) == VIEW_LVOT:
        vessel_size = int(np.clip(round(0.22 * S), 96, 160))
        context_size = int(np.clip(round(0.30 * S), 128, 192))
        c_aao = _centroid(mask, 8)
        if c_aao is not None:
            proposals.append((c_aao[0], c_aao[1], vessel_size))

        pts_lv_aao = _adjacency_points(mask, 2, 8)
        if pts_lv_aao.size > 0:
            cy = float(pts_lv_aao[:, 0].mean())
            cx = float(pts_lv_aao[:, 1].mean())
            proposals.append((cy, cx, vessel_size))

            c_lv = _centroid(mask, 2)
            if c_lv is not None and c_aao is not None:
                cy2 = float(0.5 * (c_lv[0] + c_aao[0]))
                cx2 = float(0.5 * (c_lv[1] + c_aao[1]))
                proposals.append((cy2, cx2, context_size))

    if int(view_id) == VIEW_3VT:
        vessel_size = int(np.clip(round(0.22 * S), 96, 160))
        context_size = int(np.clip(round(0.30 * S), 128, 192))
        cents = []
        for lab in (13, 9, 12):  # ARCH, MPA, SVC
            c = _centroid(mask, lab)
            if c is not None:
                cents.append(c)
                proposals.append((c[0], c[1], vessel_size))
        if len(cents) >= 2:
            cy = float(np.mean([c[0] for c in cents]))
            cx = float(np.mean([c[1] for c in cents]))
            proposals.append((cy, cx, context_size))
        elif len(cents) == 0:
            pts = np.argwhere(mask01 > 0)
            if pts.size > 0:
                y_mid = int(h * 0.55)
                pts_up = pts[pts[:, 0] < y_mid]
                src = pts_up if pts_up.size > 0 else pts
                for _ in range(2):
                    p = src[random.randrange(len(src))]
                    proposals.append((float(p[0]), float(p[1]), vessel_size))

    ek = int(erode_kernel)
    if ek % 2 == 0:
        ek += 1
    ek = max(3, ek)
    ek = min(ek, int(min(h, w) // 2 * 2 + 1))
    mask_core = binary_erode(mask01, k=ek, iters=int(erode_iters))
    coords = np.argwhere(mask_core > 0)
    if coords.size == 0:
        coords = np.argwhere(mask01 > 0)

    centers: List[Tuple[float, float, int]] = []

    for cy, cx, sz in proposals:
        if len(centers) >= locals_total:
            break
        if not all((cx - ox) ** 2 + (cy - oy) ** 2 >= (min_dist * 0.7) ** 2 for oy, ox, _ in centers):
            continue
        centers.append((cy, cx, int(sz)))

    if coords.size > 0 and locals_greedy > 0:
        coords_list = coords.tolist()
        random.shuffle(coords_list)
        for cy, cx in coords_list:
            if len(centers) >= locals_greedy:
                break
            if not all((cx - ox) ** 2 + (cy - oy) ** 2 >= min_dist ** 2 for oy, ox, _ in centers):
                continue

            local_size = base_local_size
            half = local_size // 2
            sx = int(np.floor(cx - half))
            sy = int(np.floor(cy - half))
            ex = sx + local_size
            ey = sy + local_size

            sx_clamped = max(0, sx)
            sy_clamped = max(0, sy)
            ex_clamped = min(w, ex)
            ey_clamped = min(h, ey)
            overlap = mask01[sy_clamped:ey_clamped, sx_clamped:ex_clamped]
            overlap_area = float((overlap > 0).sum())
            patch_area = float(local_size * local_size)
            if patch_area > 0 and (overlap_area / patch_area) < float(min_overlap_ratio):
                continue

            centers.append((float(cy), float(cx), 0))

    while len(centers) < locals_total:
        cx = random.uniform(x1, x2)
        cy = random.uniform(y1, y2)
        centers.append((cy, cx, 0))

    denom_x = float(max(w - 1, 1))
    denom_y = float(max(h - 1, 1))

    # NOTE: keep original mapping; if you want stricter [0,1], change /4 -> /3.
    view_norm = float(int(view_id) - 1) / 4.0

    use_hist = (mask_hist_ids is not None) and (len(mask_hist_ids) > 0)

    patches: List[np.ndarray] = []
    pos_list: List[List[float]] = []
    mhist_list: List[np.ndarray] = []

    for cy, cx, sz_override in centers[:locals_total]:
        local_size = int(sz_override) if int(sz_override) > 0 else base_local_size
        half = local_size // 2
        sx = int(np.floor(cx - half))
        sy = int(np.floor(cy - half))
        ex = sx + local_size
        ey = sy + local_size

        if use_hist:
            mcrop = _crop_and_resize_mask(mask, sx, sy, ex, ey, out_size=int(mask_hist_patch_size))
            mh_base = compute_mask_hist_ratio(mcrop, mask_hist_ids)
            mh = np.concatenate([mh_base, extra], axis=0).astype(np.float32)
        else:
            mh = np.zeros((0,), dtype=np.float32)
        mhist_list.append(mh)

        pad_left = max(0, -sx)
        pad_top = max(0, -sy)
        pad_right = max(0, ex - w)
        pad_bottom = max(0, ey - h)

        sx0 = max(0, sx)
        sy0 = max(0, sy)
        ex0 = min(w, ex)
        ey0 = min(h, ey)

        crop = img[sy0:ey0, sx0:ex0]
        if pad_left or pad_right or pad_top or pad_bottom:
            crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant")

        if crop.shape[0] != local_size or crop.shape[1] != local_size:
            pad_h = local_size - crop.shape[0]
            pad_w = local_size - crop.shape[1]
            crop = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

        if crop.shape[0] != patch_size or crop.shape[1] != patch_size:
            try:
                import cv2  # type: ignore
                crop = cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
            except Exception:
                if crop.shape[0] < patch_size or crop.shape[1] < patch_size:
                    pad_h = max(0, patch_size - crop.shape[0])
                    pad_w = max(0, patch_size - crop.shape[1])
                    crop = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
                crop = crop[:patch_size, :patch_size]

        patches.append(crop)

        cx_img = float(cx) / denom_x
        cy_img = float(cy) / denom_y
        cx_roi = float(cx - x1) / float(bw)
        cy_roi = float(cy - y1) / float(bh)
        s_rel = float(local_size) / float(max(h, w, 1))
        pos_list.append([cx_img, cy_img, cx_roi, cy_roi, s_rel, 0.0, view_norm])

    if use_hist:
        try:
            import cv2  # type: ignore
            mfull = cv2.resize(mask.astype(np.int32), (int(mask_hist_patch_size), int(mask_hist_patch_size)),
                               interpolation=cv2.INTER_NEAREST)
        except Exception:
            mfull = _crop_and_resize_mask(mask.astype(np.int32), 0, 0, w, h, out_size=int(mask_hist_patch_size))
        mh_g_base = compute_mask_hist_ratio(mfull, mask_hist_ids)
        mh_g = np.concatenate([mh_g_base, extra], axis=0).astype(np.float32)
    else:
        mh_g = np.zeros((0,), dtype=np.float32)
    mhist_list.append(mh_g)

    if h != patch_size or w != patch_size:
        try:
            import cv2  # type: ignore
            global_patch = cv2.resize(img, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
        except Exception:
            global_patch = img
            if global_patch.shape[0] < patch_size or global_patch.shape[1] < patch_size:
                pad_h = max(0, patch_size - global_patch.shape[0])
                pad_w = max(0, patch_size - global_patch.shape[1])
                global_patch = np.pad(global_patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
            global_patch = global_patch[:patch_size, :patch_size]
    else:
        global_patch = img

    patches.append(global_patch)
    pos_list.append([0.5, 0.5, 0.5, 0.5, 1.0, 1.0, view_norm])

    mhist_arr = np.stack(mhist_list, axis=0).astype(np.float32) if use_hist else np.zeros((locals_total + 1, 0), dtype=np.float32)
    return np.stack(patches, axis=0), np.asarray(pos_list, dtype=np.float32), mhist_arr


class PatchPosFullimgDataset(Dataset):
    POS_DIM = 7

    def __init__(
        self,
        ids: Sequence[str],
        images_dir: Path,
        labels_dir: Path,
        id_to_cls: Dict[str, List[int]],
        transform: T.Compose,
        patch_size: int,
        min_dist_ratio: float,
        min_overlap_ratio: float,
        locals_total: int,
        locals_greedy: int,
        bbox_margin: float,
        erode_kernel: int,
        erode_iters: int,
        mask_cond: str,
        mask_hist_ids: List[int],
        mask_hist_patch_size: int,
        num_classes: int,
    ):
        self.ids = list(ids)
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.id_to_cls = id_to_cls
        self.transform = transform
        self.patch_size = int(patch_size)
        self.min_dist_ratio = float(min_dist_ratio)
        self.min_overlap_ratio = float(min_overlap_ratio)
        self.locals_total = int(locals_total)
        self.locals_greedy = int(locals_greedy)
        self.bbox_margin = float(bbox_margin)
        self.erode_kernel = int(erode_kernel)
        self.erode_iters = int(erode_iters)
        self.num_classes = int(num_classes)

        self.mask_cond = str(mask_cond)
        self.mask_hist_ids = list(mask_hist_ids)
        self.mask_hist_patch_size = int(mask_hist_patch_size)

        self.extra_dim = 10
        self.mask_dim = (1 + len(self.mask_hist_ids) + self.extra_dim) if (self.mask_cond == "hist" and len(self.mask_hist_ids) > 0) else 0

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        cid = self.ids[idx]
        img_path = self.images_dir / f"{cid}.h5"
        lbl_path = self.labels_dir / f"{cid}_label.h5"

        with h5py.File(img_path, "r") as f:
            img = f["image"][()]
            view_id = int(f["view"][0]) if "view" in f else VIEW_4CH

        with h5py.File(lbl_path, "r") as f:
            mask = f["mask"][()] if "mask" in f else np.zeros(img.shape[:2], dtype=np.uint8)

        patches, pos, mhist = extract_patches_with_pos_fullimg(
            img=img,
            mask=mask,
            patch_size=self.patch_size,
            view_id=view_id,
            min_dist_ratio=self.min_dist_ratio,
            min_overlap_ratio=self.min_overlap_ratio,
            locals_total=self.locals_total,
            locals_greedy=self.locals_greedy,
            bbox_margin=self.bbox_margin,
            erode_kernel=self.erode_kernel,
            erode_iters=self.erode_iters,
            mask_hist_ids=self.mask_hist_ids if self.mask_cond == "hist" else None,
            mask_hist_patch_size=self.mask_hist_patch_size,
        )

        patch_tensors: List[torch.Tensor] = []
        for pimg in patches:
            t = torch.from_numpy(pimg).permute(2, 0, 1).float() / 255.0
            t = self.transform(t)
            patch_tensors.append(t)
        patches_tensor = torch.stack(patch_tensors, dim=0)

        pos_tensor = torch.from_numpy(pos).float()
        mhist_tensor = torch.from_numpy(mhist).float()

        label = torch.zeros(self.num_classes, dtype=torch.float32)
        for c in self.id_to_cls.get(cid, []):
            label[int(c)] = 1.0

        return patches_tensor, pos_tensor, mhist_tensor, label, int(view_id)

    def get_pos_neg_indices(self) -> Tuple[List[int], List[int]]:
        pos, neg = [], []
        for i, cid in enumerate(self.ids):
            if len(self.id_to_cls.get(cid, [])) > 0:
                pos.append(i)
            else:
                neg.append(i)
        return pos, neg

    # MOD(B): per-class index map for rare forcing
    def get_pos_indices_by_class(self, class_ids: List[int]) -> Dict[int, List[int]]:
        want = {int(c) for c in class_ids}
        out: Dict[int, List[int]] = {int(c): [] for c in want}
        for i, cid in enumerate(self.ids):
            cls_list = self.id_to_cls.get(cid, [])
            for c in cls_list:
                cc = int(c)
                if cc in out:
                    out[cc].append(i)
        # drop empty
        out = {c: idxs for c, idxs in out.items() if len(idxs) > 0}
        return out


class BalancedBatchSamplerRare(torch.utils.data.Sampler[List[int]]):
    """
    Balanced batching (pos/neg) + per-class rare-positive forcing:
    - every `rare_every` batches, guarantee at least one pos sample contains a selected rare class.
    - selected rare class is sampled with inverse-frequency weights (rarer => higher probability).
    - FIX: epoch length is controlled by steps_per_epoch (prevents step explosion).
    """

    def __init__(
        self,
        pos_indices: List[int],
        neg_indices: List[int],
        rare_cls_to_indices: Dict[int, List[int]],
        batch_size: int,
        rare_every: int = 4,
        steps_per_epoch: int = 100,
    ):
        self.pos_indices = list(pos_indices)
        self.neg_indices = list(neg_indices)
        self.rare_cls_to_indices = {int(k): list(v) for k, v in rare_cls_to_indices.items()}
        self.batch_size = int(batch_size)
        self.pos_bs = self.batch_size // 2
        self.neg_bs = self.batch_size - self.pos_bs
        self.rare_every = max(int(rare_every), 1)
        self.steps_per_epoch = max(int(steps_per_epoch), 1)

        if self.pos_bs <= 0 or self.neg_bs <= 0:
            raise ValueError(f"batch_size {batch_size} too small for balanced batching.")
        if len(self.pos_indices) == 0:
            raise ValueError("No positive samples in train set: balance-batch cannot run.")
        if len(self.neg_indices) == 0:
            print("[WARN] No negative samples in train set. Sampler will draw from positives only for neg half.")

        # prepare inverse-freq weights for rare classes
        self.rare_classes: List[int] = []
        self.rare_weights: List[float] = []
        for c, idxs in sorted(self.rare_cls_to_indices.items(), key=lambda kv: kv[0]):
            if len(idxs) <= 0:
                continue
            self.rare_classes.append(int(c))
            self.rare_weights.append(1.0 / float(len(idxs)))

        # normalize weights (optional; random.choices doesn't require normalization)
        s = sum(self.rare_weights)
        if s > 0:
            self.rare_weights = [w / s for w in self.rare_weights]

    def __iter__(self):
        for b in range(self.steps_per_epoch):
            batch: List[int] = []
            force_rare = (b % self.rare_every == 0) and (len(self.rare_classes) > 0)

            if force_rare:
                # MOD(B): sample class first, then sample index from that class
                c = random.choices(self.rare_classes, weights=self.rare_weights, k=1)[0]
                batch.append(random.choice(self.rare_cls_to_indices[c]))
                remain = self.pos_bs - 1
                if remain > 0:
                    batch.extend(random.choices(self.pos_indices, k=remain))
            else:
                batch.extend(random.choices(self.pos_indices, k=self.pos_bs))

            if self.neg_indices:
                batch.extend(random.choices(self.neg_indices, k=self.neg_bs))
            else:
                batch.extend(random.choices(self.pos_indices, k=self.neg_bs))

            while len(batch) < self.batch_size:
                batch.append(random.choice(self.pos_indices))

            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.steps_per_epoch


def unfreeze_last_k_blocks(backbone: nn.Module, k: int = 4) -> None:
    for p in backbone.parameters():
        p.requires_grad_(False)

    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise AttributeError("Backbone has no attribute 'blocks'; cannot unfreeze last-k blocks.")

    if k > 0:
        for blk in blocks[-k:]:
            for p in blk.parameters():
                p.requires_grad_(True)

    for name in ["norm", "fc_norm"]:
        if hasattr(backbone, name):
            mod = getattr(backbone, name)
            if isinstance(mod, nn.Module):
                for p in mod.parameters():
                    p.requires_grad_(True)


def build_backbone_param_groups(
    backbone: nn.Module,
    base_lr: float,
    layer_decay: float,
    unfrozen_blocks: int,
) -> List[Dict[str, Any]]:
    blocks = getattr(backbone, "blocks", None)
    if blocks is None or unfrozen_blocks <= 0:
        return []

    total_blocks = len(blocks)
    if total_blocks == 0:
        return []

    unfrozen_blocks = min(int(unfrozen_blocks), total_blocks)
    start_idx = total_blocks - unfrozen_blocks
    layer_decay = float(max(min(layer_decay, 1.0), 0.0))

    groups: List[Dict[str, Any]] = []
    for idx in range(start_idx, total_blocks):
        block = blocks[idx]
        params = [p for p in block.parameters() if p.requires_grad]
        if not params:
            continue
        depth = idx - start_idx
        scale = layer_decay ** (unfrozen_blocks - 1 - depth) if unfrozen_blocks > 1 else 1.0
        groups.append({"params": params, "lr": float(base_lr) * float(scale)})

    for name in ("norm", "fc_norm"):
        mod = getattr(backbone, name, None)
        if isinstance(mod, nn.Module):
            norm_params = [p for p in mod.parameters() if p.requires_grad]
            if norm_params:
                groups.append({"params": norm_params, "lr": float(base_lr)})

    return groups


def build_cosine_warmup_scheduler(optim: torch.optim.Optimizer, total_epochs: int, warmup_epochs: int):
    total_epochs = max(int(total_epochs), 1)
    warmup_epochs = max(min(int(warmup_epochs), total_epochs), 0)

    def lr_lambda(step: int) -> float:
        if warmup_epochs > 0 and step < warmup_epochs:
            return float(step + 1) / float(warmup_epochs)
        if total_epochs == warmup_epochs:
            return 1.0
        progress = (step - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)


class DinoPatchFullimgFiLMClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        pos_dim: int = 7,
        mask_dim: int = 0,
        film_hidden: int = 64,
        film_scale_pos: float = 0.02,
        film_scale_mask: float = 0.005,
        mask_cond_dropout: float = 0.5,
        local_scale: float = 0.05,   # kept for compatibility
        global_scale: float = 0.2,   # kept for compatibility
        head_hidden: int = 256,
        head_dropout: float = 0.1,
        head_layernorm: bool = True,
        # NEW (vote):
        vote_clip: float = 4.0,
        vote_lam: float = 0.35,
        out_scale: float = 1.0,
        wL_init: float = 0.8,
        # MOD(A)
        mix_bias_max: float = 0.5,
    ):
        super().__init__()
        self.backbone = backbone
        self.pos_dim = int(pos_dim)
        self.mask_dim = int(mask_dim)
        self.film_scale_pos = float(film_scale_pos)
        self.film_scale_mask = float(film_scale_mask)
        self.mask_cond_dropout = float(mask_cond_dropout)

        # keep old scalars for logging/compat
        self.local_scale = float(local_scale)
        self.global_scale = float(global_scale)

        # NEW vote params
        self.vote_clip = float(vote_clip)
        self.vote_lam = float(vote_lam)
        self.out_scale = float(out_scale)

        # MOD(A): bounded bias
        self.mix_bias_max = float(mix_bias_max)

        self.pos_mlp = nn.Sequential(
            nn.Linear(self.pos_dim, int(film_hidden)),
            nn.GELU(),
            nn.Linear(int(film_hidden), 2 * 768),
        )
        nn.init.zeros_(self.pos_mlp[-1].weight)
        nn.init.zeros_(self.pos_mlp[-1].bias)

        if self.mask_dim > 0:
            self.mask_mlp = nn.Sequential(
                nn.Linear(self.mask_dim, int(film_hidden)),
                nn.GELU(),
                nn.Linear(int(film_hidden), 2 * 768),
            )
            nn.init.zeros_(self.mask_mlp[-1].weight)
            nn.init.zeros_(self.mask_mlp[-1].bias)
        else:
            self.mask_mlp = None

        layers: List[nn.Module] = [nn.Linear(768, int(head_hidden))]
        if head_layernorm:
            layers.append(nn.LayerNorm(int(head_hidden)))
        layers += [
            nn.GELU(),
            nn.Dropout(float(head_dropout)),
            nn.Linear(int(head_hidden), int(num_classes)),
        ]
        self.classifier = nn.Sequential(*layers)

        # NEW: per-class stable mixing weights (locals vs global)
        wL_init = float(np.clip(float(wL_init), 1e-4, 1.0 - 1e-4))
        init_logit = math.log(wL_init / (1.0 - wL_init))
        self.mix_logit = nn.Parameter(torch.full((int(num_classes),), float(init_logit)))

        # MOD(A): raw bias (unbounded) -> bounded via tanh in forward
        self.bias_raw = nn.Parameter(torch.zeros(int(num_classes)))

    def forward(self, patches: torch.Tensor, pos: torch.Tensor, mhist: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, c, h, w = patches.shape
        if pos.shape[0] != b or pos.shape[1] != n or pos.shape[2] != self.pos_dim:
            raise ValueError(f"pos shape {tuple(pos.shape)} incompatible with patches {tuple(patches.shape)}")

        flat = patches.view(b * n, c, h, w)
        feats = self.backbone.forward_features(flat)["x_norm_clstoken"]

        pos_flat = pos.view(b * n, self.pos_dim)
        gb_pos = self.pos_mlp(pos_flat)
        g_pos, b_pos = gb_pos.chunk(2, dim=-1)
        gamma = 1.0 + self.film_scale_pos * torch.tanh(g_pos)
        beta = self.film_scale_pos * torch.tanh(b_pos)

        if self.mask_mlp is not None:
            if mhist is None:
                raise ValueError("mask_mlp enabled but mhist is None")
            if mhist.shape[0] != b or mhist.shape[1] != n or mhist.shape[2] != self.mask_dim:
                raise ValueError(f"mhist shape {tuple(mhist.shape)} incompatible, expected (B,N,{self.mask_dim})")

            mh = mhist.view(b * n, self.mask_dim)
            p = self.mask_cond_dropout
            if self.training and p > 0:
                keep = (torch.rand((mh.shape[0], 1), device=mh.device) > p).float()
                mh = mh * keep / (1.0 - p)

            gb_m = self.mask_mlp(mh)
            g_m, b_m = gb_m.chunk(2, dim=-1)

            gamma = gamma + self.film_scale_mask * torch.tanh(g_m)
            beta = beta + self.film_scale_mask * torch.tanh(b_m)

        feats_mod = gamma * feats + beta
        logits = self.classifier(feats_mod).view(b, n, -1)

        locals_ = logits[:, :-1, :]      # (B,L,K)
        global_ = logits[:, -1, :]       # (B,K)

        # NEW vote: Tanh-Clip Mean + Disagreement Penalty
        c = float(self.vote_clip)
        c = max(c, 1e-6)
        loc = c * torch.tanh(locals_ / c)                       # (B,L,K)
        m = loc.mean(dim=1)                                     # (B,K)
        s = loc.std(dim=1, unbiased=False)                      # (B,K)
        local_cons = m - float(self.vote_lam) * s               # (B,K)

        wL = torch.sigmoid(self.mix_logit).unsqueeze(0)         # (1,K)

        # MOD(A): bounded bias
        if self.mix_bias_max > 0:
            bias = float(self.mix_bias_max) * torch.tanh(self.bias_raw).unsqueeze(0)  # (1,K)
        else:
            bias = 0.0

        out = float(self.out_scale) * (wL * local_cons + (1.0 - wL) * global_) + bias
        return out


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    film_scale_pos = float(args.film_scale_pos) if args.film_scale_pos is not None else float(args.film_scale)

    script_dir = Path(__file__).resolve().parent
    conversion_script = script_dir / "convert_pretraining_nnUNet_to_h5.py"
    nnunet_dataset_root = args.nnunet_raw_root / args.nnunet_dataset_name

    images_dir = args.images
    labels_dir = args.labels

    if args.nnunet_plan:
        run_nnunet_plan_and_preprocess(args.nnunet_task_id, args.nnunet_raw_root, args.nnunet_preprocessed_root, args.nnunet_verify)

    if args.convert_nnunet_to_h5:
        convert_pretraining_to_h5(nnunet_dataset_root, args.nnunet_h5_root, conversion_script, args.nnunet_convert_classes)
        images_dir = args.nnunet_h5_root / "images"
        labels_dir = args.nnunet_h5_root / "labels"

    for pth in [images_dir, labels_dir, args.weights, args.disease_json]:
        if not Path(pth).exists():
            raise FileNotFoundError(f"Path not found: {pth}")

    print(f"Using images from: {images_dir}")
    print(f"Using labels from: {labels_dir}")
    print(f"Output directory: {args.output_dir}")

    all_ids = collect_all_ids(images_dir, labels_dir)  # cid -> view_id (1..4)
    disease_cases = load_disease_cases(args.disease_json)
    if not disease_cases:
        raise ValueError(f"Disease JSON {args.disease_json} defines no classes.")
    num_classes = len(disease_cases)

    cls_allowed = load_cls_allowed(args.cls_allowed, DEFAULT_CLS_ALLOWED)
    print(f"cls_allowed (view_id=1..4): {cls_allowed}")

    mask_hist_ids = _parse_csv_list(args.mask_hist_ids, int) if args.mask_cond == "hist" else []
    print(f"[mask_cond] mode={args.mask_cond}, ids={mask_hist_ids}, extra_dim=10, dropout={args.mask_cond_dropout}")
    print(f"[film] scale_pos={film_scale_pos}, scale_mask={args.film_scale_mask}, hidden={args.film_hidden}")
    print(f"[mix_bias] max={args.mix_bias_max} (bias = max*tanh(raw); set 0 to disable)")

    test_ids: List[str] = []
    if args.test_json is not None:
        if not args.test_json.exists():
            raise FileNotFoundError(f"test_json not found: {args.test_json}")
        test_ids = load_ids_from_json(args.test_json)

    test_set = set(test_ids)
    if args.test_json is not None and (not args.merge_test_json_into_train):
        all_ids = {cid: v for cid, v in all_ids.items() if cid not in test_set}

    pos_ids, id_to_cls = build_pos_labels(disease_cases)
    pos_ids = [cid for cid in pos_ids if cid in all_ids]
    id_to_cls = {cid: cls_list for cid, cls_list in id_to_cls.items() if cid in all_ids}
    pos_set = set(pos_ids)

    split_ids = all_ids
    if args.merge_test_json_into_train and test_set:
        split_ids = {cid: v for cid, v in all_ids.items() if cid not in test_set}

    split_output_dir = args.split_json_dir or (args.output_dir / "splits")
    load_split_dir = args.load_split_json_dir
    print(f"Split JSON directory: {split_output_dir}")
    if load_split_dir is not None:
        print(f"Loading splits from: {load_split_dir}")

    labels_all: Dict[str, List[int]] = {cid: id_to_cls.get(cid, []) for cid in pos_ids}

    def _normalize_ids(raw: List) -> List[str]:
        normalized: List[str] = []
        for entry in raw:
            if isinstance(entry, dict):
                cid = entry.get("case_id") or entry.get("id") or entry.get("name")
                if cid is None:
                    raise ValueError(f"Split entry missing case identifier: {entry}")
                normalized.append(str(cid))
            else:
                normalized.append(str(entry))
        return normalized

    if load_split_dir is not None:
        def _load_split(name: str) -> List[str]:
            path = load_split_dir / f"{name}.json"
            if not path.exists():
                raise FileNotFoundError(f"Split file not found: {path}")
            with path.open("r") as f:
                data = json.load(f)
            return _normalize_ids(data)

        train_ids_raw = _load_split("train")
        val_ids_raw = _load_split("val")
        internal_test_ids = _load_split("test")
    else:
        class_members = compute_class_members(split_ids, disease_cases)
        split_id_to_cls = {cid: cls_list for cid, cls_list in id_to_cls.items() if cid in split_ids}
        pos_set_split = set(split_id_to_cls.keys())
        train_pos_ids, val_pos_ids, test_pos_ids = stratified_split_positive_ids(split_id_to_cls, class_members, args.seed)
        neg_candidates = [cid for cid in sorted(split_ids) if cid not in pos_set_split]
        neg_train_ids, neg_val_ids, neg_test_ids = split_negatives_evenly(neg_candidates, args.seed)

        train_ids_raw = train_pos_ids + neg_train_ids
        val_ids_raw = val_pos_ids + neg_val_ids
        internal_test_ids = test_pos_ids + neg_test_ids

        write_split_json(split_output_dir / "train.json", train_ids_raw)
        write_split_json(split_output_dir / "val.json", val_ids_raw)
        write_split_json(split_output_dir / "test.json", internal_test_ids)

    if args.merge_test_json_into_train and test_ids:
        merge_ids = [cid for cid in test_ids if cid in all_ids]
        if merge_ids:
            merge_set = set(merge_ids)
            train_ids_raw = list(dict.fromkeys(list(train_ids_raw) + merge_ids))
            val_ids_raw = [cid for cid in val_ids_raw if cid not in merge_set]
            internal_test_ids = [cid for cid in internal_test_ids if cid not in merge_set]
            print(f"[split] merged {len(merge_ids)} test_json ids into train")
        else:
            print("[split] merge-test-json-into-train enabled but no test_json ids found in dataset")

    # ===== Keep your requirement: merge internal test into val =====
    if internal_test_ids:
        val_ids_raw = list(dict.fromkeys(val_ids_raw + internal_test_ids))
        internal_test_ids = []

    for cid in set(train_ids_raw + val_ids_raw + internal_test_ids):
        labels_all.setdefault(cid, id_to_cls.get(cid, []))

    # ===== pos_weight under view-masked counting =====
    pos_weight = compute_pos_weight_viewmasked(
        train_ids=train_ids_raw,
        labels_all=labels_all,
        all_views=all_ids,
        cls_allowed=cls_allowed,
        num_classes=num_classes,
        cap=args.pos_weight_cap,
    ).to(device)
    print(f"pos_weight (view-masked sqrt neg/pos, capped): {pos_weight.detach().cpu().numpy().round(3).tolist()}")

    transform = build_transform(args.img_size)

    train_ds = PatchPosFullimgDataset(
        ids=train_ids_raw,
        images_dir=images_dir,
        labels_dir=labels_dir,
        id_to_cls=labels_all,
        transform=transform,
        patch_size=args.patch_size,
        min_dist_ratio=args.min_dist_ratio,
        min_overlap_ratio=args.min_overlap_ratio,
        locals_total=args.locals_total,
        locals_greedy=args.locals_greedy,
        bbox_margin=args.bbox_margin,
        erode_kernel=args.erode_kernel,
        erode_iters=args.erode_iters,
        mask_cond=args.mask_cond,
        mask_hist_ids=mask_hist_ids,
        mask_hist_patch_size=args.mask_hist_patch_size,
        num_classes=num_classes,
    )
    val_ds = PatchPosFullimgDataset(
        ids=val_ids_raw,
        images_dir=images_dir,
        labels_dir=labels_dir,
        id_to_cls=labels_all,
        transform=transform,
        patch_size=args.patch_size,
        min_dist_ratio=args.min_dist_ratio,
        min_overlap_ratio=args.min_overlap_ratio,
        locals_total=args.locals_total,
        locals_greedy=args.locals_greedy,
        bbox_margin=args.bbox_margin,
        erode_kernel=args.erode_kernel,
        erode_iters=args.erode_iters,
        mask_cond=args.mask_cond,
        mask_hist_ids=mask_hist_ids,
        mask_hist_patch_size=args.mask_hist_patch_size,
        num_classes=num_classes,
    )

    if args.balance_batch:
        t_pos, t_neg = train_ds.get_pos_neg_indices()
        rare_classes = _parse_csv_list(args.rare_classes, int)

        # MOD(B): build per-class map
        rare_cls_map = train_ds.get_pos_indices_by_class(rare_classes)

        if args.steps_per_epoch and args.steps_per_epoch > 0:
            steps_per_epoch = int(args.steps_per_epoch)
            steps_msg = "manual"
        else:
            steps_per_epoch = int(math.ceil(len(train_ds) / float(max(args.batch_size, 1))))
            steps_msg = "auto"

        rare_counts = {c: len(idxs) for c, idxs in rare_cls_map.items()}
        print(
            f"[batch] balance-batch ON | pos={len(t_pos)} neg={len(t_neg)} "
            f"| rare_every={args.rare_every} rare_classes={rare_classes} rare_counts={rare_counts} "
            f"| steps_per_epoch={steps_per_epoch} ({steps_msg})"
        )

        train_sampler = BalancedBatchSamplerRare(
            pos_indices=t_pos,
            neg_indices=t_neg,
            rare_cls_to_indices=rare_cls_map,
            batch_size=args.batch_size,
            rare_every=args.rare_every,
            steps_per_epoch=steps_per_epoch,
        )
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # aggregation scaling (kept exactly as before)
    if args.local_scale is not None and float(args.local_scale) >= 0:
        local_scale = float(args.local_scale)
        auto_msg = "manual"
    else:
        local_scale = float(args.local_total_weight) / float(max(1, args.locals_total))
        auto_msg = "auto"
    global_scale = float(args.global_scale)
    print(f"[agg] locals_total={args.locals_total} local_scale={local_scale:.6f} ({auto_msg}, local_total_weight={args.local_total_weight}) | global_scale={global_scale:.6f}")

    # NEW: init vote mixing using your old effective weights (so behavior starts close)
    local_eff = float(args.locals_total) * float(local_scale)
    global_eff = float(global_scale)
    wL_init = local_eff / (local_eff + global_eff + 1e-6)
    out_scale = local_eff + global_eff
    print(f"[vote] clip={args.vote_clip} lam={args.vote_lam} | init_wL={wL_init:.4f} out_scale={out_scale:.4f}")

    backbone = dinov3_vitb16(weights=str(args.weights), pretrained=True)
    model = DinoPatchFullimgFiLMClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pos_dim=PatchPosFullimgDataset.POS_DIM,
        mask_dim=train_ds.mask_dim,
        film_hidden=args.film_hidden,
        film_scale_pos=film_scale_pos,
        film_scale_mask=float(args.film_scale_mask) if args.mask_cond == "hist" else 0.0,
        mask_cond_dropout=float(args.mask_cond_dropout),
        local_scale=local_scale,
        global_scale=global_scale,
        head_hidden=256,
        head_dropout=0.1,
        head_layernorm=True,
        vote_clip=float(args.vote_clip),
        vote_lam=float(args.vote_lam),
        out_scale=float(out_scale),
        wL_init=float(wL_init),
        mix_bias_max=float(args.mix_bias_max),
    ).to(device)

    loss_fn = AsymmetricLossMultiLabel(
        gamma_pos=args.asym_gamma_pos,
        gamma_neg=args.asym_gamma_neg,
        clip=args.asym_clip,
        eps=args.asym_eps,
        pos_weight=pos_weight,
    ).to(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best_fc.pt"
    last_path = args.output_dir / "last_fc.pt"

    # =========================
    # Eval with FIXED F1 calc
    # (F1 LOGIC UNCHANGED)
    # =========================
    def eval_loader(loader):
        model.eval()
        total_loss = 0.0
        K = num_classes
        tp = torch.zeros(K, device=device)
        fp = torch.zeros(K, device=device)
        fn = torch.zeros(K, device=device)

        with torch.no_grad():
            for patches, pos, mhist, target, view_id in loader:
                patches = patches.to(device, non_blocking=True)
                pos = pos.to(device, non_blocking=True)
                mhist = mhist.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                views = torch.as_tensor(view_id, dtype=torch.int64, device=device)

                logits = model(patches, pos, mhist if args.mask_cond == "hist" else None)

                allowed = build_allowed_mask_torch(views, cls_allowed, num_classes, device=device)  # (B,K)

                loss_elem = loss_fn(logits, target, reduction="none")  # (B,K)
                denom = allowed.sum().clamp_min(1.0)
                loss = (loss_elem * allowed).sum() / denom
                total_loss += loss.item() * patches.size(0)

                pred = (torch.sigmoid(logits) > 0.5).float()

                tp += (pred * target * allowed).sum(dim=0)
                fp += (pred * (1.0 - target) * allowed).sum(dim=0)
                fn += ((1.0 - pred) * target * allowed).sum(dim=0)

        denom_f = (2 * tp + fp + fn).clamp_min(1e-8)
        per_class_f1 = (2 * tp) / denom_f

        appear = (tp + fn) > 0
        if appear.any():
            macro_f1 = per_class_f1[appear].mean().item()
        else:
            macro_f1 = 0.0

        return total_loss / len(loader.dataset), macro_f1

    def _load_checkpoint(path: Path) -> None:
        ckpt = torch.load(path, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint {path}")

    def _run_stage(
        stage_idx: int,
        stage_epochs: int,
        unfreeze_blocks: int,
        head_lr: float,
        backbone_lr: float,
        load_ckpt: Optional[Path],
    ) -> Path:
        stage_tag = f"stage{stage_idx + 1}_k{unfreeze_blocks}"
        stage_best = args.output_dir / f"best_fc_{stage_tag}.pt"
        stage_last = args.output_dir / f"last_fc_{stage_tag}.pt"

        unfreeze_last_k_blocks(model.backbone, k=unfreeze_blocks)
        if load_ckpt is not None:
            _load_checkpoint(load_ckpt)

        backbone_groups = build_backbone_param_groups(
            model.backbone,
            backbone_lr,
            args.layer_decay,
            unfreeze_blocks,
        )
        head_group = {"params": model.classifier.parameters(), "lr": head_lr}
        pos_group = {"params": model.pos_mlp.parameters(), "lr": head_lr}

        # MOD(A): include bias_raw + mix_logit
        mix_group = {"params": [model.mix_logit, model.bias_raw], "lr": head_lr}

        groups = backbone_groups + [head_group, pos_group, mix_group]
        if model.mask_mlp is not None:
            groups.append({"params": model.mask_mlp.parameters(), "lr": head_lr})

        optim = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        sched = build_cosine_warmup_scheduler(optim, stage_epochs, args.warmup_epochs)

        best_val_f1 = float("-inf")
        no_improve = 0

        for epoch in range(1, stage_epochs + 1):
            model.train()
            tot_loss = 0.0

            for patches, pos, mhist, target, view_id in train_loader:
                patches = patches.to(device, non_blocking=True)
                pos = pos.to(device, non_blocking=True)
                mhist = mhist.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                views = torch.as_tensor(view_id, dtype=torch.int64, device=device)

                logits = model(patches, pos, mhist if args.mask_cond == "hist" else None)

                allowed = build_allowed_mask_torch(views, cls_allowed, num_classes, device=device)  # (B,K)
                loss_elem = loss_fn(logits, target, reduction="none")  # (B,K)
                denom = allowed.sum().clamp_min(1.0)
                loss = (loss_elem * allowed).sum() / denom

                optim.zero_grad()
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optim.step()

                tot_loss += loss.item() * patches.size(0)

            train_loss = tot_loss / len(train_loader.dataset)
            val_loss, val_f1 = eval_loader(val_loader)

            lr_backbone_min, lr_backbone_max = None, None
            if len(backbone_groups) > 0:
                backbone_lrs = [pg["lr"] for pg in optim.param_groups[:len(backbone_groups)]]
                lr_backbone_min = min(backbone_lrs)
                lr_backbone_max = max(backbone_lrs)
            head_lr_now = optim.param_groups[len(backbone_groups)]["lr"]

            if lr_backbone_min is not None:
                print(
                    f"[{stage_tag}] Epoch {epoch:03d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                    f"val_f1 {val_f1:.4f} | lr_backbone [{lr_backbone_min:.6f},{lr_backbone_max:.6f}] | lr_head {head_lr_now:.6f}"
                )
            else:
                print(
                    f"[{stage_tag}] Epoch {epoch:03d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                    f"val_f1 {val_f1:.4f} | lr_head {head_lr_now:.6f}"
                )

            torch.save({"state_dict": model.state_dict(), "pos_weight": pos_weight, "epoch": epoch}, stage_last)
            torch.save({"state_dict": model.state_dict(), "pos_weight": pos_weight, "epoch": epoch}, last_path)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({"state_dict": model.state_dict(), "pos_weight": pos_weight, "epoch": epoch}, stage_best)
                torch.save({"state_dict": model.state_dict(), "pos_weight": pos_weight, "epoch": epoch}, best_path)
                print(f"Saved best to {stage_best}")
                no_improve = 0
            else:
                no_improve += 1

            if args.early_stop_patience and no_improve >= args.early_stop_patience:
                print(f"[{stage_tag}] Early stopping after {no_improve} epochs without improvement.")
                break

            sched.step()

        return stage_best

    stage_unfreeze = _parse_csv_list(args.stage_unfreeze, int)
    stage_epochs_list = _parse_csv_list(args.stage_epochs, int)
    stage_head_lrs = _parse_csv_list(args.stage_head_lrs, float)
    stage_backbone_lrs = _parse_csv_list(args.stage_backbone_lrs, float)

    if stage_unfreeze:
        print(f"Staged unfreeze enabled: {stage_unfreeze}")
        prev_best: Optional[Path] = None
        for i, k in enumerate(stage_unfreeze):
            stage_epochs = int(_get_stage_value(stage_epochs_list, i, args.epochs))
            stage_head_lr = float(_get_stage_value(stage_head_lrs, i, args.lr))
            stage_backbone_lr = float(_get_stage_value(stage_backbone_lrs, i, args.backbone_lr))

            load_ckpt = prev_best
            if i == 0 and args.init_checkpoint is not None:
                load_ckpt = args.init_checkpoint

            prev_best = _run_stage(
                stage_idx=i,
                stage_epochs=stage_epochs,
                unfreeze_blocks=int(k),
                head_lr=stage_head_lr,
                backbone_lr=stage_backbone_lr,
                load_ckpt=load_ckpt,
            )
    else:
        _run_stage(
            stage_idx=0,
            stage_epochs=args.epochs,
            unfreeze_blocks=args.unfreeze_blocks,
            head_lr=args.lr,
            backbone_lr=args.backbone_lr,
            load_ckpt=args.init_checkpoint if args.init_checkpoint is not None else None,
        )


if __name__ == "__main__":
    main()

"""
# Example command (same as yours; add --mix-bias-max if you want tighter bias)
PYTHONPATH=. python /gpfs/work/aac/haoyuwu24/dinov3/train_fc_patch_fullimg_film_v2_localweight_rarepatch_revise_vote.py \
  --weights /gpfs/work/aac/haoyuwu24/dinov3/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --images /gpfs/work/aac/haoyuwu24/images \
  --labels /gpfs/work/aac/haoyuwu24/labels \
  --disease-json /gpfs/work/aac/haoyuwu24/cls_index_find/disease_cases.json \
  --output-dir /gpfs/work/aac/haoyuwu24/dinov3/outputs/FULL_pos7_hist_rel_vote_v2 \
  --epochs 200 --batch-size 8 --num-workers 1 \
  --unfreeze-blocks 4 --backbone-lr 2e-5 --lr 3e-4 --layer-decay 0.75 --warmup-epochs 10 \
  --locals-total 8 --locals-greedy 8 --bbox-margin 0.15 --min-overlap-ratio 0.10 --min-dist-ratio 0.5 \
  --mask-cond hist --mask-hist-ids "1,2,3,4,5,6,7,8,9,12,13" --mask-hist-patch-size 64 --mask-cond-dropout 0.5 \
  --film-hidden 64 --film-scale-pos 0.02 --film-scale-mask 0.005 \
  --vote-clip 4.0 --vote-lam 0.35 \
  --balance-batch --rare-classes "1,2,3" --rare-every 4 \
  --mix-bias-max 0.3
"""
