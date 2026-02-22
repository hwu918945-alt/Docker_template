#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed-parameter inference entrypoint for the submission Docker.

Default I/O:
  - input images: /input/images  (H5 files)
  - output dir:   /output

Environment overrides:
  INPUT_DIR, OUTPUT_DIR, TMP_DIR
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    app_root = Path("/workspace")
    input_dir = Path(os.environ.get("INPUT_DIR", "/input/images"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/output"))
    tmp_dir = Path(os.environ.get("TMP_DIR", "/tmp/val_infer_unet"))

    seg_ckpt = app_root / "weights/seg/best.pth"
    cls_ckpt = app_root / "weights/cls/best_fc_stage1_k4.pt"
    dino_weights = app_root / "weights/dino/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    disease_json = app_root / "cls_index_find/disease_cases.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(app_root / "bsaeline/FETUS-Challenge-ISBI2026/run_val_inference_unet_chd_combined.py"),
        "--val-images",
        str(input_dir),
        "--out-dir",
        str(output_dir),
        "--tmp-dir",
        str(tmp_dir),
        "--seg-ckpt",
        str(seg_ckpt),
        "--cls-script",
        str(app_root / "dinov3/infer_full_pos7_hist_rel_v2_revise_vote_v2.py"),
        "--cls-ckpt",
        str(cls_ckpt),
        "--cls-weights",
        str(dino_weights),
        "--cls-disease-json",
        str(disease_json),
        "--cls-locals-total",
        "8",
        "--cls-locals-greedy",
        "8",
        "--cls-min-overlap-ratio",
        "0.10",
        "--cls-min-dist-ratio",
        "0.5",
        "--cls-bbox-margin",
        "0.15",
        "--cls-erode-kernel",
        "5",
        "--cls-erode-iters",
        "1",
        "--cls-local-total-weight",
        "0.8",
        "--cls-local-scale",
        "-1",
        "--cls-global-scale",
        "0.2",
        "--cls-film-hidden",
        "64",
        "--cls-film-scale-pos",
        "0.02",
        "--cls-film-scale-mask",
        "0.005",
        "--cls-mask-cond",
        "hist",
        "--cls-mask-cond-dropout",
        "0.5",
        "--cls-mask-hist-ids",
        "1,2,3,4,5,6,7,8,9,12,13",
        "--cls-mask-hist-patch-size",
        "64",
        "--cls-vote-clip",
        "4.0",
        "--cls-vote-lam",
        "0.35",
        "--cls-thresholds",
        "[0.5,0.5,0.5,0.5,0.5,0.5,0.5]",
    ]

    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f\"{app_root}/dinov3:{app_root}/bsaeline/FETUS-Challenge-ISBI2026{':' + py_path if py_path else ''}\"

    print(\"Running:\", \" \".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=str(app_root / \"dinov3\"))
    return 0


if __name__ == \"__main__\":
    raise SystemExit(main())
