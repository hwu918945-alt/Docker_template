import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parent


DEF_EXTRA_ARGS = "--cls-locals-total 16 --cls-locals-greedy 16 --cls-min-overlap-ratio 0.10 --cls-min-dist-ratio 0.5 --cls-bbox-margin 0.15 --cls-erode-kernel 5 --cls-erode-iters 1 --cls-local-total-weight 0.8 --cls-local-scale -1 --cls-global-scale 0.2 --cls-film-hidden 64 --cls-film-scale-pos 0.02 --cls-film-scale-mask 0.005 --cls-mask-cond hist --cls-mask-cond-dropout 0.5 --cls-mask-hist-ids 1,2,3,4,5,6,7,8,9,12,13 --cls-mask-hist-patch-size 64 --cls-vote-clip 4.0 --cls-vote-lam 0.35 --cls-thresholds \"[0.5,0.5,0.5,0.5,0.5,0.5,0.5]\""


def _default_path(env_name: str, rel_path: str) -> Path:
    val = os.environ.get(env_name)
    if val:
        return Path(val)
    return REPO_ROOT / rel_path


def _load_data_json(path: Path) -> List[Path]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("--data-json must be a list")
    images: List[Path] = []
    for item in data:
        if isinstance(item, str):
            images.append(Path(item))
        elif isinstance(item, dict):
            img = item.get("image") or item.get("image_path")
            if not img:
                raise ValueError("Entry missing 'image' field")
            images.append(Path(img))
        else:
            raise ValueError(f"Unsupported entry in data-json: {type(item)}")
    return images


def _prepare_inputs(images: List[Path], tmp_dir: Path) -> tuple[Path, Path]:
    parents = {p.parent.resolve() for p in images}
    if len(parents) == 1:
        images_dir = parents.pop()
        case_ids = [p.stem for p in images]
    else:
        images_dir = tmp_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        case_ids = []
        for idx, src in enumerate(images):
            cid = src.stem
            if cid in case_ids:
                cid = f"{cid}_{idx}"
            dst = images_dir / f"{cid}.h5"
            shutil.copy2(src, dst)
            case_ids.append(cid)
    case_json = tmp_dir / "case_list.json"
    case_json.parent.mkdir(parents=True, exist_ok=True)
    with case_json.open("w", encoding="utf-8") as f:
        json.dump([{"case_id": cid} for cid in case_ids], f, indent=2)
    return images_dir, case_json


def parse_args():
    p = argparse.ArgumentParser("FETUS2026 inference wrapper")
    p.add_argument("--data-json", required=True, help="JSON list of input H5 files")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--tmp-dir", default=os.environ.get("TMP_DIR", "/tmp/val_infer_unet"))

    p.add_argument("--seg-ckpt", default=str(_default_path("SEG_CKPT", "prepare/best.pth")))
    p.add_argument("--cls-script", default=str(_default_path("CLS_SCRIPT", "prepare/infer_full_pos7_hist_rel_v2_revise_vote_v2.py")))
    p.add_argument("--cls-ckpt", default=str(_default_path("CLS_CKPT", "prepare/best_fc.pt")))
    p.add_argument("--cls-weights", default=str(_default_path("CLS_WEIGHTS", "prepare/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")))
    p.add_argument("--cls-disease-json", default=str(_default_path("CLS_DISEASE_JSON", "cls_index_find/disease_cases.json")))
    p.add_argument("--extra-args", default=os.environ.get("EXTRA_ARGS", DEF_EXTRA_ARGS))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_json = Path(args.data_json)
    out_dir = Path(args.out_dir)
    tmp_dir = Path(args.tmp_dir)

    if not data_json.exists():
        raise FileNotFoundError(f"data-json not found: {data_json}")

    images = _load_data_json(data_json)
    for p in images:
        if not p.exists():
            raise FileNotFoundError(f"image not found: {p}")

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    images_dir, case_json = _prepare_inputs(images, tmp_dir)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "prepare/run_val_inference_unet_chd_combined.py"),
        "--images",
        str(images_dir),
        "--json",
        str(case_json),
        "--out-dir",
        str(out_dir),
        "--tmp-dir",
        str(tmp_dir),
        "--seg-ckpt",
        str(args.seg_ckpt),
        "--cls-script",
        str(args.cls_script),
        "--cls-ckpt",
        str(args.cls_ckpt),
        "--cls-weights",
        str(args.cls_weights),
        "--cls-disease-json",
        str(args.cls_disease_json),
    ]

    if args.extra_args:
        cmd += args.extra_args.split()

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



