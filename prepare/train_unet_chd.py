#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standard UNet training for CHD segmentation stored in .h5 files.
Each .h5 is expected to have: image (H,W,3 uint8), mask (H,W uint8), view (1,) int64.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None

try:
    from torch.amp import autocast as _autocast
    from torch.amp import GradScaler as _GradScaler
    _AMP_SOURCE = "torch.amp"
except Exception:  # pragma: no cover
    from torch.cuda.amp import autocast as _autocast
    from torch.cuda.amp import GradScaler as _GradScaler
    _AMP_SOURCE = "torch.cuda.amp"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_float_list(s: str, n: int) -> Tuple[float, ...]:
    vals = [float(x) for x in s.split(",") if x.strip()]
    if len(vals) == 1:
        vals = vals * n
    if len(vals) != n:
        raise ValueError(f"Expected {n} floats, got {len(vals)}")
    return tuple(vals)


def parse_int_list(s: str, n: int) -> Tuple[int, ...]:
    s = s.lower().replace("x", ",")
    vals = [int(x) for x in s.split(",") if x.strip()]
    if len(vals) == 1:
        vals = vals * n
    if len(vals) != n:
        raise ValueError(f"Expected {n} ints, got {len(vals)}")
    return tuple(vals)


def load_index_json(index_json: Path) -> List[dict]:
    with index_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError("index.json must be a list of dicts")
    return obj


def build_ids_from_index(
    index_json: Path,
    train_split: str,
    val_split: str,
    include_empty_in_train: bool,
) -> Tuple[List[str], List[str]]:
    obj = load_index_json(index_json)
    train_ids: List[str] = []
    val_ids: List[str] = []
    for o in obj:
        split = str(o.get("split", ""))
        new_id = str(o["new_id"])
        if split == train_split or (include_empty_in_train and split == ""):
            train_ids.append(new_id)
        elif split == val_split:
            val_ids.append(new_id)
    return train_ids, val_ids


def list_h5_ids(data_dir: Path) -> List[str]:
    ids = [p.stem for p in sorted(data_dir.glob("*.h5"))]
    if not ids:
        raise FileNotFoundError(f"No .h5 files found in {data_dir}")
    return ids


def random_split(ids: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    ids = list(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(round(len(ids) * val_ratio))
    return ids[n_val:], ids[:n_val]


def read_h5_pair(h5_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        img = f["image"][:]
        mask = f["mask"][:]
    return img, mask


def infer_label_range(
    data_dir: Path,
    ids: List[str],
    cache_dir: Optional[Path],
    stop_if_exceeds: Optional[int],
) -> Tuple[int, int]:
    minv = 1_000_000_000
    maxv = -1
    for idx in ids:
        if cache_dir is not None:
            p = cache_dir / f"{idx}.npz"
            if p.exists():
                with np.load(p) as npz:
                    m = npz["mask"]
            else:
                _, m = read_h5_pair(data_dir / f"{idx}.h5")
        else:
            _, m = read_h5_pair(data_dir / f"{idx}.h5")
        minv = min(minv, int(m.min()))
        maxv = max(maxv, int(m.max()))
        if stop_if_exceeds is not None and maxv >= stop_if_exceeds:
            break
    return minv, maxv


def to_tensor(img: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    if img.ndim == 2:
        img = img[:, :, None]
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    img_t = torch.from_numpy(img).permute(2, 0, 1)  # CHW
    mask_t = torch.from_numpy(mask.astype(np.int64))  # HW
    return img_t, mask_t


def build_preprocess_cache(
    data_dir: Path,
    ids: List[str],
    cache_dir: Path,
    overwrite: bool,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    total = len(ids)
    for i, idx in enumerate(ids, 1):
        out_path = cache_dir / f"{idx}.npz"
        if out_path.exists() and not overwrite:
            continue
        img, mask = read_h5_pair(data_dir / f"{idx}.h5")
        img_t, mask_t = to_tensor(img, mask)
        np.savez_compressed(out_path, img=img_t.numpy(), mask=mask_t.numpy())
        if i == 1 or i % 50 == 0 or i == total:
            print(f"[preprocess] {i}/{total} cached", flush=True)


def resolve_torch_class(path: Optional[str]) -> Optional[type]:
    if not path:
        return None
    name = path.split(".")[-1]
    if hasattr(nn, name):
        return getattr(nn, name)
    raise ValueError(f"Unsupported torch class: {path}")


def load_nnunet_plan(plan_path: Path, config: str) -> Dict[str, Any]:
    obj = json.loads(plan_path.read_text())
    if "configurations" not in obj:
        raise ValueError(f"Invalid plan file: {plan_path}")
    if config not in obj["configurations"]:
        raise ValueError(f"Config '{config}' not found in plan {plan_path}")
    return obj["configurations"][config]


class PolyLRScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9) -> None:
        self.optimizer = optimizer
        self.initial_lr = float(initial_lr)
        self.max_steps = max_steps
        self.exponent = float(exponent)
        self._last_lr = [group["lr"] for group in optimizer.param_groups]

    def step(self, current_step: int) -> None:
        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def get_last_lr(self) -> List[float]:
        return self._last_lr


_PLOT_WARNED = False


def append_metrics_csv(path: Path, header: List[str], row: List[Any]) -> None:
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(header)
        writer.writerow(row)


def save_plots(plot_dir: Path, history: Dict[str, List[float]]) -> None:
    global _PLOT_WARNED
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        if not _PLOT_WARNED:
            print("Warning: matplotlib not installed; skipping PNG plots.", flush=True)
            _PLOT_WARNED = True
        return

    plot_dir.mkdir(parents=True, exist_ok=True)
    epochs = history.get("epoch", [])
    if not epochs:
        return

    def plot_series(keys: List[str], title: str, fname: str) -> None:
        plt.figure(figsize=(7, 4))
        for k in keys:
            if k in history and len(history[k]) == len(epochs):
                plt.plot(epochs, history[k], label=k)
        plt.title(title)
        plt.xlabel("epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / fname, dpi=150)
        plt.close()

    # Single-figure overlay: loss + dice + lr (lr on right axis)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        if "train_loss" in history:
            ax.plot(epochs, history["train_loss"], label="train_loss", color="tab:blue")
        if "val_dice" in history:
            ax.plot(epochs, history["val_dice"], label="val_dice", color="tab:green")
        if "mean_fg_dice" in history:
            ax.plot(epochs, history["mean_fg_dice"], label="mean_fg_dice", color="tab:olive")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss / dice")
        ax2 = ax.twinx()
        if "lr" in history:
            ax2.plot(epochs, history["lr"], label="lr", color="tab:red")
        ax2.set_ylabel("lr")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")
        plt.tight_layout()
        plt.savefig(plot_dir / "loss_dice_lr.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass

    plot_series(["val_iou", "mean_fg_iou"], "IoU", "iou.png")
    plot_series(["mean_fg_precision", "mean_fg_recall"], "Precision/Recall (FG)", "precision_recall.png")
    plot_series(["mean_fg_specificity"], "Specificity (FG)", "specificity.png")
    plot_series(["acc"], "Accuracy", "accuracy.png")

def apply_affine_2d(
    img: torch.Tensor,
    mask: torch.Tensor,
    angle_rad: float,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if angle_rad == 0.0 and scale == 1.0:
        return img, mask
    ca = float(np.cos(angle_rad))
    sa = float(np.sin(angle_rad))
    theta = torch.tensor(
        [[scale * ca, -scale * sa, 0.0], [scale * sa, scale * ca, 0.0]],
        dtype=img.dtype,
        device=img.device,
    )
    grid = F.affine_grid(theta.unsqueeze(0), size=(1, img.shape[0], img.shape[1], img.shape[2]), align_corners=False)
    img_out = F.grid_sample(img.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False).squeeze(0)
    mask_out = (
        F.grid_sample(mask[None, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)
        .squeeze(0)
        .squeeze(0)
        .long()
    )
    return img_out, mask_out


def gaussian_blur_2d(img: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=img.dtype, device=img.device)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    c = img.shape[0]
    img_b = img.unsqueeze(0)
    k_h = kernel.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    k_w = kernel.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    img_b = F.conv2d(img_b, k_h, padding=(radius, 0), groups=c)
    img_b = F.conv2d(img_b, k_w, padding=(0, radius), groups=c)
    return img_b.squeeze(0)


def gamma_transform(img: torch.Tensor, gamma: float, invert: bool, retain_stats: bool) -> torch.Tensor:
    out = img.clone()
    for c in range(out.shape[0]):
        x = out[c]
        if invert:
            x = x * -1.0
        if retain_stats:
            mean = torch.mean(x)
            std = torch.std(x).clamp_min(1e-7)
        minm = torch.min(x)
        rnge = torch.max(x) - minm
        x = torch.pow((x - minm) / torch.clamp(rnge, min=1e-7), gamma) * rnge + minm
        if retain_stats:
            mn_here = torch.mean(x)
            std_here = torch.std(x).clamp_min(1e-7)
            x = (x - mn_here) * (std / std_here) + mean
        if invert:
            x = x * -1.0
        out[c] = x
    return out

class CHDH5Dataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        ids: List[str],
        img_size: int,
        patch_size: Optional[Tuple[int, int]],
        random_crop: bool,
        augment: bool,
        aug_mode: str,
        hflip: float,
        vflip: float,
        rot90: bool,
        brightness: float,
        contrast: float,
        noise_std: float,
        normalize: str,
        mean: Optional[Tuple[float, float, float]],
        std: Optional[Tuple[float, float, float]],
        cache_dir: Optional[Path],
    ) -> None:
        self.data_dir = data_dir
        self.ids = ids
        self.img_size = img_size
        self.patch_size = patch_size
        self.random_crop = random_crop
        self.augment = augment
        self.aug_mode = aug_mode
        self.hflip = hflip
        self.vflip = vflip
        self.rot90 = rot90
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.cache_dir = cache_dir
        if self.patch_size is not None:
            h, w = self.patch_size
        elif self.img_size > 0:
            h, w = self.img_size, self.img_size
        else:
            h, w = 256, 256
        ratio = max(h, w) / max(1, min(h, w))
        rot_deg = 15.0 if ratio > 1.5 else 180.0
        self.rotation_rad = float(np.deg2rad(rot_deg))

    def __len__(self) -> int:
        return len(self.ids)

    def _load_h5(self, idx: str) -> Tuple[np.ndarray, np.ndarray]:
        p = self.data_dir / f"{idx}.h5"
        return read_h5_pair(p)

    def _to_tensor(self, img: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        return to_tensor(img, mask)

    def _load_cache(self, idx: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache_dir is None:
            raise RuntimeError("cache_dir is not set")
        p = self.cache_dir / f"{idx}.npz"
        if not p.exists():
            raise FileNotFoundError(f"Cache miss: {p}. Run with --preprocess_dir to build cache.")
        with np.load(p) as npz:
            img = npz["img"]
            mask = npz["mask"]
        img_t = torch.from_numpy(img.astype(np.float32))
        mask_t = torch.from_numpy(mask.astype(np.int64))
        return img_t, mask_t

    def _crop_or_pad(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.patch_size is None:
            return img, mask
        target_h, target_w = self.patch_size
        _, h, w = img.shape
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), value=0.0)
            mask = F.pad(mask, (0, pad_w, 0, pad_h), value=0)
            _, h, w = img.shape
        if h == target_h and w == target_w:
            return img, mask
        if self.random_crop:
            top = random.randint(0, h - target_h)
            left = random.randint(0, w - target_w)
        else:
            top = max(0, (h - target_h) // 2)
            left = max(0, (w - target_w) // 2)
        img = img[:, top : top + target_h, left : left + target_w]
        mask = mask[top : top + target_h, left : left + target_w]
        return img, mask

    def _resize(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.img_size <= 0:
            return img, mask
        _, h, w = img.shape
        if h == self.img_size and w == self.img_size:
            return img, mask
        img = F.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size), mode="bilinear", align_corners=False).squeeze(0)
        mask = F.interpolate(mask[None, None].float(), size=(self.img_size, self.img_size), mode="nearest").squeeze(0).squeeze(0).long()
        return img, mask

    def _augment_basic(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.hflip:
            img = torch.flip(img, dims=(2,))
            mask = torch.flip(mask, dims=(1,))
        if random.random() < self.vflip:
            img = torch.flip(img, dims=(1,))
            mask = torch.flip(mask, dims=(0,))
        if self.rot90:
            k = random.randint(0, 3)
            if k:
                img = torch.rot90(img, k, dims=(1, 2))
                mask = torch.rot90(mask, k, dims=(0, 1))
        if self.contrast > 0:
            alpha = 1.0 + (random.random() * 2.0 - 1.0) * self.contrast
            img = img * alpha
        if self.brightness > 0:
            beta = (random.random() * 2.0 - 1.0) * self.brightness
            img = img + beta
        if self.noise_std > 0:
            img = img + torch.randn_like(img) * self.noise_std
        img = img.clamp(0.0, 1.0)
        return img, mask

    def _augment_nnunet(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        angle = 0.0
        scale = 1.0
        do_affine = False
        if random.random() < 0.2:
            angle = random.uniform(-self.rotation_rad, self.rotation_rad)
            do_affine = True
        if random.random() < 0.2:
            scale = random.uniform(0.7, 1.4)
            do_affine = True
        if do_affine:
            img, mask = apply_affine_2d(img, mask, angle, scale)

        if random.random() < 0.1:
            var = random.uniform(0.0, 0.1)
            if var > 0:
                noise = torch.randn(1, img.shape[1], img.shape[2], dtype=img.dtype, device=img.device) * np.sqrt(var)
                img = img + noise

        if random.random() < 0.2:
            sigma = random.uniform(0.5, 1.0)
            img = gaussian_blur_2d(img, sigma)

        if random.random() < 0.15:
            mult = torch.empty((img.shape[0], 1, 1), dtype=img.dtype, device=img.device).uniform_(0.75, 1.25)
            img = img * mult

        if random.random() < 0.15:
            mean = img.mean(dim=(1, 2), keepdim=True)
            factor = torch.empty((img.shape[0], 1, 1), dtype=img.dtype, device=img.device).uniform_(0.75, 1.25)
            img = (img - mean) * factor + mean

        if random.random() < 0.25:
            scale_lr = random.uniform(0.5, 1.0)
            h, w = img.shape[1], img.shape[2]
            new_h = max(1, int(round(h * scale_lr)))
            new_w = max(1, int(round(w * scale_lr)))
            img_lr = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
            img = F.interpolate(img_lr.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)

        if random.random() < 0.1:
            gamma = random.uniform(0.7, 1.5)
            img = gamma_transform(img, gamma, invert=True, retain_stats=True)
        if random.random() < 0.3:
            gamma = random.uniform(0.7, 1.5)
            img = gamma_transform(img, gamma, invert=False, retain_stats=True)

        if random.random() < 0.5:
            img = torch.flip(img, dims=(2,))
            mask = torch.flip(mask, dims=(1,))
        if random.random() < 0.5:
            img = torch.flip(img, dims=(1,))
            mask = torch.flip(mask, dims=(0,))

        img = img.clamp(0.0, 1.0)
        return img, mask

    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        if self.normalize == "none":
            return img
        if self.normalize == "per_image":
            mean = img.mean(dim=(1, 2), keepdim=True)
            std = img.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            return (img - mean) / std
        if self.normalize == "mean_std":
            if self.mean is None or self.std is None:
                raise ValueError("mean/std must be provided when normalize=mean_std")
            mean = torch.tensor(self.mean, dtype=img.dtype, device=img.device)[:, None, None]
            std = torch.tensor(self.std, dtype=img.dtype, device=img.device)[:, None, None]
            return (img - mean) / std
        raise ValueError(f"Unknown normalize={self.normalize}")

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache_dir is not None:
            img_t, mask_t = self._load_cache(self.ids[i])
        else:
            img, mask = self._load_h5(self.ids[i])
            img_t, mask_t = self._to_tensor(img, mask)
        if self.augment:
            if self.aug_mode == "nnunet":
                img_t, mask_t = self._augment_nnunet(img_t, mask_t)
            else:
                img_t, mask_t = self._augment_basic(img_t, mask_t)
        if self.patch_size is not None:
            img_t, mask_t = self._crop_or_pad(img_t, mask_t)
        else:
            img_t, mask_t = self._resize(img_t, mask_t)
        img_t = self._normalize(img_t)
        return img_t, mask_t


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, bilinear: bool) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = DoubleConv(in_ch + skip_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base_channels: int, bilinear: bool) -> None:
        super().__init__()
        self.in_conv = DoubleConv(in_channels, base_channels)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels, base_channels * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels * 2, base_channels * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels * 4, base_channels * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels * 8, base_channels * 16))

        self.up1 = Up(base_channels * 16, base_channels * 8, base_channels * 8, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4, base_channels * 4, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2, base_channels * 2, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, base_channels, bilinear)

        self.out_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out_conv(x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_convs: int,
        conv_op: type,
        conv_bias: bool,
        norm_op: Optional[type],
        norm_op_kwargs: Optional[dict],
        nonlin: Optional[type],
        nonlin_kwargs: Optional[dict],
        kernel_size: Tuple[int, int],
        first_stride: Tuple[int, int],
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        stride = first_stride
        for i in range(n_convs):
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            conv = conv_op(
                in_ch if i == 0 else out_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=conv_bias,
            )
            layers.append(conv)
            if norm_op is not None:
                layers.append(norm_op(out_ch, **(norm_op_kwargs or {})))
            if nonlin is not None:
                layers.append(nonlin(**(nonlin_kwargs or {})))
            stride = (1, 1)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PlanUNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, arch_kwargs: Dict[str, Any]) -> None:
        super().__init__()
        n_stages = int(arch_kwargs["n_stages"])
        features = [int(x) for x in arch_kwargs["features_per_stage"]]
        n_conv_per_stage = [int(x) for x in arch_kwargs["n_conv_per_stage"]]
        n_conv_per_stage_decoder = [int(x) for x in arch_kwargs["n_conv_per_stage_decoder"]]
        strides = [tuple(s) for s in arch_kwargs["strides"]]
        kernel_sizes = [tuple(k) for k in arch_kwargs["kernel_sizes"]]

        conv_op = resolve_torch_class(arch_kwargs.get("conv_op")) or nn.Conv2d
        norm_op = resolve_torch_class(arch_kwargs.get("norm_op"))
        norm_op_kwargs = arch_kwargs.get("norm_op_kwargs", {})
        nonlin = resolve_torch_class(arch_kwargs.get("nonlin"))
        nonlin_kwargs = arch_kwargs.get("nonlin_kwargs", {})
        conv_bias = bool(arch_kwargs.get("conv_bias", True))

        self.down_blocks = nn.ModuleList()
        for s in range(n_stages):
            in_ch = in_channels if s == 0 else features[s - 1]
            out_ch = features[s]
            block = ConvBlock(
                in_ch=in_ch,
                out_ch=out_ch,
                n_convs=n_conv_per_stage[s],
                conv_op=conv_op,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                kernel_size=kernel_sizes[s],
                first_stride=strides[s],
            )
            self.down_blocks.append(block)

        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i, s in enumerate(range(n_stages - 1, 0, -1)):
            stride = strides[s]
            self.up_convs.append(
                nn.ConvTranspose2d(features[s], features[s - 1], kernel_size=stride, stride=stride)
            )
            block = ConvBlock(
                in_ch=features[s - 1] * 2,
                out_ch=features[s - 1],
                n_convs=n_conv_per_stage_decoder[i],
                conv_op=conv_op,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                kernel_size=kernel_sizes[s - 1],
                first_stride=(1, 1),
            )
            self.up_blocks.append(block)

        self.out_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        for block in self.down_blocks:
            x = block(x)
            skips.append(x)
        for i, (up, block) in enumerate(zip(self.up_convs, self.up_blocks)):
            skip = skips[-(i + 2)]
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = block(x)
        return self.out_conv(x)


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    include_bg: bool,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    targets_oh = F.one_hot(targets.clamp_min(0), num_classes=num_classes).permute(0, 3, 1, 2).float()
    if ignore_index >= 0:
        mask = (targets != ignore_index).unsqueeze(1)
        probs = probs * mask
        targets_oh = targets_oh * mask
    dims = (0, 2, 3)
    inter = (probs * targets_oh).sum(dims)
    denom = probs.sum(dims) + targets_oh.sum(dims)
    dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
    if not include_bg and num_classes > 1:
        dice = dice[1:]
    return 1.0 - dice.mean()


class ConfusionMeter:
    def __init__(self, num_classes: int, ignore_index: int) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.cm = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.view(-1).to(torch.int64)
        target = target.view(-1).to(torch.int64)
        if self.ignore_index >= 0:
            mask = target != self.ignore_index
            pred = pred[mask]
            target = target[mask]
        k = self.num_classes * target + pred
        cm = torch.bincount(k, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        self.cm += cm.cpu()

    def compute(self, include_bg: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        inter = torch.diag(self.cm)
        pred_sum = self.cm.sum(0)
        target_sum = self.cm.sum(1)
        dice = (2.0 * inter) / (pred_sum + target_sum + 1e-6)
        iou = inter / (pred_sum + target_sum - inter + 1e-6)
        precision = inter / (pred_sum + 1e-6)
        recall = inter / (target_sum + 1e-6)
        total = self.cm.sum()
        tn = total - (pred_sum + target_sum - inter)
        specificity = tn / (tn + (pred_sum - inter) + 1e-6)
        acc = float(inter.sum() / (total + 1e-6))
        if not include_bg and self.num_classes > 1:
            dice = dice[1:]
            iou = iou[1:]
            precision = precision[1:]
            recall = recall[1:]
            specificity = specificity[1:]
        return dice, iou, precision, recall, specificity, acc


def compute_class_weights(
    data_dir: Path,
    ids: List[str],
    num_classes: int,
) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for idx in ids:
        p = data_dir / f"{idx}.h5"
        with h5py.File(p, "r") as f:
            mask = f["mask"][:].astype(np.int64)
        bc = torch.bincount(torch.from_numpy(mask.reshape(-1)), minlength=num_classes).to(torch.float64)
        counts += bc
    freq = counts / counts.sum().clamp_min(1.0)
    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.mean()
    return weights.float()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    include_bg: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, float, float, float]:
    model.eval()
    meter = ConfusionMeter(num_classes=num_classes, ignore_index=ignore_index)
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)
        meter.update(preds, masks)
    dice, iou, precision, recall, specificity, acc = meter.compute(include_bg=include_bg)
    mean_dice = float(dice.mean().item()) if dice.numel() > 0 else 0.0
    mean_iou = float(iou.mean().item()) if iou.numel() > 0 else 0.0
    if include_bg:
        fg_dice = dice[1:] if dice.numel() > 1 else dice
        fg_iou = iou[1:] if iou.numel() > 1 else iou
        fg_precision = precision[1:] if precision.numel() > 1 else precision
        fg_recall = recall[1:] if recall.numel() > 1 else recall
        fg_specificity = specificity[1:] if specificity.numel() > 1 else specificity
    else:
        fg_dice = dice
        fg_iou = iou
        fg_precision = precision
        fg_recall = recall
        fg_specificity = specificity
    mean_fg_dice = float(fg_dice.mean().item()) if fg_dice.numel() > 0 else 0.0
    mean_fg_iou = float(fg_iou.mean().item()) if fg_iou.numel() > 0 else 0.0
    mean_fg_precision = float(fg_precision.mean().item()) if fg_precision.numel() > 0 else 0.0
    mean_fg_recall = float(fg_recall.mean().item()) if fg_recall.numel() > 0 else 0.0
    mean_fg_specificity = float(fg_specificity.mean().item()) if fg_specificity.numel() > 0 else 0.0
    return (
        dice,
        iou,
        precision,
        recall,
        specificity,
        acc,
        mean_dice,
        mean_iou,
        mean_fg_dice,
        mean_fg_iou,
        mean_fg_precision,
        mean_fg_recall,
        mean_fg_specificity,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--index_json", type=Path, default=None)
    p.add_argument("--train_split", type=str, default="Tr")
    p.add_argument("--val_split", type=str, default="Ts")
    p.add_argument("--include_empty_in_train", action="store_true")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--patch_size", type=str, default=None, help="HxW or single int (e.g. 512 or 512,512)")
    p.add_argument("--num_classes", type=int, default=0, help="0 = auto from masks")
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--bilinear", action="store_true")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--momentum", type=float, default=0.99)
    p.add_argument("--nesterov", action="store_true")
    p.add_argument("--no-nesterov", dest="nesterov", action="store_false")
    p.add_argument("--poly_exp", type=float, default=0.9)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=Path, required=True)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--amp", dest="amp", action="store_true")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--log_dir", type=Path, default=None)

    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "step", "plateau", "poly"])
    p.add_argument("--step_size", type=int, default=100)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--plateau_patience", type=int, default=20)
    p.add_argument("--plateau_factor", type=float, default=0.5)

    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--hflip", type=float, default=0.5)
    p.add_argument("--vflip", type=float, default=0.0)
    p.add_argument("--rot90", action="store_true")
    p.add_argument("--brightness", type=float, default=0.1)
    p.add_argument("--contrast", type=float, default=0.1)
    p.add_argument("--noise_std", type=float, default=0.0)
    p.add_argument("--aug_mode", type=str, default="basic", choices=["basic", "nnunet"])

    p.add_argument("--normalize", type=str, default="per_image", choices=["per_image", "mean_std", "none"])
    p.add_argument("--mean", type=str, default="0.5,0.5,0.5")
    p.add_argument("--std", type=str, default="0.5,0.5,0.5")

    p.add_argument("--class_weights", type=str, default="none", choices=["none", "inv_freq"])
    p.add_argument("--ce_weight", type=float, default=1.0)
    p.add_argument("--dice_weight", type=float, default=1.0)
    p.add_argument("--dice_include_bg", action="store_true")
    p.add_argument("--ignore_index", type=int, default=-1)
    p.add_argument("--best_metric", type=str, default="dice", choices=["dice", "iou"])
    p.add_argument("--preprocess_dir", type=Path, default=None)
    p.add_argument("--preprocess_overwrite", action="store_true")
    p.add_argument("--preprocess_only", action="store_true")
    p.add_argument("--nnunet_plan", type=Path, default=None)
    p.add_argument("--nnunet_plan_config", type=str, default="2d")
    p.add_argument("--no_plan_override", dest="plan_override", action="store_false")
    p.add_argument("--no_plan_arch", dest="use_plan_arch", action="store_false")
    p.add_argument("--preset", type=str, default="default", choices=["default", "nnunet"])

    p.set_defaults(amp=True, plan_override=True, use_plan_arch=True, nesterov=True)

    args = p.parse_args()
    set_seed(args.seed)

    data_dir = args.data_dir
    index_json = args.index_json or (data_dir / "index.json")
    if index_json.exists():
        train_ids, val_ids = build_ids_from_index(
            index_json=index_json,
            train_split=args.train_split,
            val_split=args.val_split,
            include_empty_in_train=args.include_empty_in_train,
        )
    else:
        all_ids = list_h5_ids(data_dir)
        train_ids, val_ids = random_split(all_ids, args.val_ratio, args.seed)

    if not train_ids or not val_ids:
        raise ValueError(f"Empty split: train={len(train_ids)} val={len(val_ids)}")

    patch_size: Optional[Tuple[int, int]] = None
    if args.patch_size:
        patch_size = parse_int_list(args.patch_size, 2)

    plan_arch: Optional[Dict[str, Any]] = None
    if args.nnunet_plan is not None:
        plan_cfg = load_nnunet_plan(args.nnunet_plan, args.nnunet_plan_config)
        plan_patch_size = tuple(plan_cfg["patch_size"])
        plan_batch_size = int(plan_cfg["batch_size"])
        plan_norm = plan_cfg.get("normalization_schemes", [])
        plan_arch = plan_cfg.get("architecture", {}).get("arch_kwargs")
        if args.plan_override:
            args.batch_size = plan_batch_size
            patch_size = plan_patch_size
            if any("ZScore" in str(n) for n in plan_norm):
                args.normalize = "per_image"
        if args.preset == "default":
            args.preset = "nnunet"

    if args.preset == "nnunet":
        args.epochs = 1000
        args.lr = 1e-2
        args.weight_decay = 3e-5
        args.optimizer = "sgd"
        args.scheduler = "poly"
        args.momentum = 0.99
        args.nesterov = True
        args.aug_mode = "nnunet"

    if args.preprocess_only and args.preprocess_dir is None:
        raise ValueError("--preprocess_only requires --preprocess_dir")

    if args.preprocess_dir is not None:
        print("[stage] preprocess cache", flush=True)
        all_ids = sorted(set(train_ids + val_ids))
        build_preprocess_cache(
            data_dir=args.data_dir,
            ids=all_ids,
            cache_dir=args.preprocess_dir,
            overwrite=args.preprocess_overwrite,
        )
        if args.preprocess_only:
            print(f"[done] preprocess cache written to {args.preprocess_dir}")
            return

    if args.num_classes <= 0:
        minv, maxv = infer_label_range(args.data_dir, train_ids + val_ids, args.preprocess_dir, None)
        args.num_classes = maxv + 1
        print(f"[info] auto num_classes={args.num_classes} (label range {minv}..{maxv})", flush=True)
    else:
        minv, maxv = infer_label_range(args.data_dir, train_ids + val_ids, args.preprocess_dir, args.num_classes)
        if minv < 0 or maxv >= args.num_classes:
            raise ValueError(
                f"Label out of range: min={minv} max={maxv} for num_classes={args.num_classes}. "
                "Set --num_classes to max_label+1 or fix masks."
            )

    mean = parse_float_list(args.mean, 3) if args.normalize == "mean_std" else None
    std = parse_float_list(args.std, 3) if args.normalize == "mean_std" else None

    train_ds = CHDH5Dataset(
        data_dir=data_dir,
        ids=train_ids,
        img_size=args.img_size,
        patch_size=patch_size,
        random_crop=patch_size is not None,
        augment=not args.no_augment,
        aug_mode=args.aug_mode,
        hflip=args.hflip,
        vflip=args.vflip,
        rot90=args.rot90,
        brightness=args.brightness,
        contrast=args.contrast,
        noise_std=args.noise_std,
        normalize=args.normalize,
        mean=mean,
        std=std,
        cache_dir=args.preprocess_dir,
    )
    val_ds = CHDH5Dataset(
        data_dir=data_dir,
        ids=val_ids,
        img_size=args.img_size,
        patch_size=patch_size,
        random_crop=False,
        augment=False,
        aug_mode="basic",
        hflip=0.0,
        vflip=0.0,
        rot90=False,
        brightness=0.0,
        contrast=0.0,
        noise_std=0.0,
        normalize=args.normalize,
        mean=mean,
        std=std,
        cache_dir=args.preprocess_dir,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if plan_arch is not None and args.use_plan_arch:
        model = PlanUNet(
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            arch_kwargs=plan_arch,
        ).to(device)
    else:
        model = UNet(
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            base_channels=args.base_channels,
            bilinear=args.bilinear,
        ).to(device)

    class_weights = None
    if args.class_weights == "inv_freq":
        class_weights = compute_class_weights(data_dir, train_ids, args.num_classes).to(device)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=args.nesterov,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.min_lr,
            verbose=True,
        )
    elif args.scheduler == "poly":
        scheduler = PolyLRScheduler(optimizer, initial_lr=args.lr, max_steps=args.epochs, exponent=args.poly_exp)
    amp_enabled = args.amp and device.type == "cuda"
    if _AMP_SOURCE == "torch.amp":
        scaler = _GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=amp_enabled)
    else:
        scaler = _GradScaler(enabled=amp_enabled)

    start_epoch = 1
    best_score = -1.0
    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir or (save_dir / "tb")
    if SummaryWriter is None:
        print("Warning: tensorboard not installed; disabling TensorBoard logging.")
        writer = None
    else:
        writer = SummaryWriter(log_dir=str(log_dir))
    metrics_csv = save_dir / "metrics.csv"
    plot_dir = save_dir / "plots"
    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_dice": [],
        "val_iou": [],
        "mean_fg_dice": [],
        "mean_fg_iou": [],
        "mean_fg_precision": [],
        "mean_fg_recall": [],
        "mean_fg_specificity": [],
        "acc": [],
        "lr": [],
        "epoch_time_sec": [],
        "elapsed_sec": [],
    }
    train_start_time = time.time()

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and amp_enabled:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_score = float(ckpt.get("best_score", -1.0))

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        print(f"[stage] train epoch {epoch}/{args.epochs}", flush=True)
        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast("cuda" if device.type == "cuda" else "cpu", enabled=amp_enabled):
                logits = model(imgs)
                ce = F.cross_entropy(logits, masks, weight=class_weights, ignore_index=args.ignore_index)
                dl = dice_loss(logits, masks, args.num_classes, args.ignore_index, args.dice_include_bg)
                loss = args.ce_weight * ce + args.dice_weight * dl
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item())
            n_batches += 1

        mean_loss = epoch_loss / max(n_batches, 1)
        print(f"[stage] val epoch {epoch}/{args.epochs}", flush=True)
        (
            dice,
            iou,
            precision,
            recall,
            specificity,
            acc,
            val_dice,
            val_iou,
            mean_fg_dice,
            mean_fg_iou,
            mean_fg_precision,
            mean_fg_recall,
            mean_fg_specificity,
        ) = evaluate(
            model, val_loader, device, args.num_classes, args.ignore_index, args.dice_include_bg
        )
        score = val_dice if args.best_metric == "dice" else val_iou
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start_time
        print(
            f"[epoch {epoch}] loss={mean_loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f} "
            f"mean_fg_dice={mean_fg_dice:.4f} mean_fg_iou={mean_fg_iou:.4f} "
            f"mean_fg_prec={mean_fg_precision:.4f} mean_fg_rec={mean_fg_recall:.4f} "
            f"mean_fg_spec={mean_fg_specificity:.4f} acc={acc:.4f} time={epoch_time:.1f}s",
            flush=True,
        )
        dice_list = [float(x) for x in dice.cpu().numpy().tolist()] if dice.numel() > 0 else []
        iou_list = [float(x) for x in iou.cpu().numpy().tolist()] if iou.numel() > 0 else []
        prec_list = [float(x) for x in precision.cpu().numpy().tolist()] if precision.numel() > 0 else []
        rec_list = [float(x) for x in recall.cpu().numpy().tolist()] if recall.numel() > 0 else []
        spec_list = [float(x) for x in specificity.cpu().numpy().tolist()] if specificity.numel() > 0 else []
        print(f"[metrics] dice_per_class={dice_list}", flush=True)
        print(f"[metrics] iou_per_class={iou_list}", flush=True)
        print(f"[metrics] precision_per_class={prec_list}", flush=True)
        print(f"[metrics] recall_per_class={rec_list}", flush=True)
        print(f"[metrics] specificity_per_class={spec_list}", flush=True)
        if writer is not None:
            writer.add_scalar("train/loss", mean_loss, epoch)
            writer.add_scalar("val/dice", val_dice, epoch)
            writer.add_scalar("val/iou", val_iou, epoch)
            writer.add_scalar("val/mean_fg_precision", mean_fg_precision, epoch)
            writer.add_scalar("val/mean_fg_recall", mean_fg_recall, epoch)
            writer.add_scalar("val/mean_fg_specificity", mean_fg_specificity, epoch)
            writer.add_scalar("val/acc", acc, epoch)
            for i, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"lr/group{i}", pg["lr"], epoch)

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(score)
            elif args.scheduler == "poly":
                scheduler.step(epoch)
            else:
                scheduler.step()

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "args": vars(args),
        }

        if score > best_score:
            best_score = score
            ckpt["best_score"] = best_score
            torch.save(ckpt, save_dir / "best.pth")
            print(f"[save] best.pth ({args.best_metric}={best_score:.4f})", flush=True)

        lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(float(epoch))
        history["train_loss"].append(mean_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["mean_fg_dice"].append(mean_fg_dice)
        history["mean_fg_iou"].append(mean_fg_iou)
        history["mean_fg_precision"].append(mean_fg_precision)
        history["mean_fg_recall"].append(mean_fg_recall)
        history["mean_fg_specificity"].append(mean_fg_specificity)
        history["acc"].append(acc)
        history["lr"].append(lr)
        history["epoch_time_sec"].append(epoch_time)
        history["elapsed_sec"].append(elapsed)

        header = [
            "epoch",
            "train_loss",
            "val_dice",
            "val_iou",
            "mean_fg_dice",
            "mean_fg_iou",
            "mean_fg_precision",
            "mean_fg_recall",
            "mean_fg_specificity",
            "acc",
            "lr",
            "epoch_time_sec",
            "elapsed_sec",
        ]
        header += [f"dice_c{i}" for i in range(len(dice_list))]
        header += [f"iou_c{i}" for i in range(len(iou_list))]
        header += [f"prec_c{i}" for i in range(len(prec_list))]
        header += [f"recall_c{i}" for i in range(len(rec_list))]
        header += [f"spec_c{i}" for i in range(len(spec_list))]
        row = [
            epoch,
            mean_loss,
            val_dice,
            val_iou,
            mean_fg_dice,
            mean_fg_iou,
            mean_fg_precision,
            mean_fg_recall,
            mean_fg_specificity,
            acc,
            lr,
            epoch_time,
            elapsed,
        ] + dice_list + iou_list + prec_list + rec_list + spec_list
        append_metrics_csv(metrics_csv, header, row)
        save_plots(plot_dir, history)

    torch.save(ckpt, save_dir / "last.pth")
    print(f"Best {args.best_metric}={best_score:.4f} saved to {save_dir / 'best.pth'}")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
