"""Restore a single image with a trained checkpoint.

python -m src.predict --checkpoint outputs/checkpoints/amfm_final/best_psnr.pth \
    --input path/to/degraded.png --output restored.png
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .evaluate import tiled_predict
from .model import build_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="restored.png")
    ap.add_argument("--tile", type=int, default=512)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(state["config"]["model"]).to(device).eval()
    model.load_state_dict(state["model"])

    img = np.asarray(Image.open(args.input).convert("RGB"), dtype=np.float32) / 255.0
    x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        y = tiled_predict(model, x, tile=args.tile).clamp(0, 1)
    arr = (y[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(args.output)
    print(f"saved {args.output}  ({arr.shape[1]}x{arr.shape[0]})")


if __name__ == "__main__":
    main()
