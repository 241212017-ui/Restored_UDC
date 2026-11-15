"""Full-resolution evaluation on POLED / TOLED test (or val) splits."""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .data import UDCMatDataset
from .metrics import DISTSMetric, LPIPSMetric, psnr, ssim
from .model import build_model
from .utils import count_parameters_m, describe_device, get_device, save_json


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    return (x.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


def _perceptual_pair(pred, gt, max_side):
    if max_side and max(pred.shape[-2:]) > max_side:
        scale = max_side / max(pred.shape[-2:])
        size = (int(round(pred.shape[-2] * scale)), int(round(pred.shape[-1] * scale)))
        pred = F.interpolate(pred, size=size, mode="bilinear", align_corners=False, antialias=True)
        gt = F.interpolate(gt, size=size, mode="bilinear", align_corners=False, antialias=True)
    return pred, gt


@torch.no_grad()
def tiled_predict(model, lq: torch.Tensor, tile: int = 512, overlap: int = 64) -> torch.Tensor:
    """Memory-safe fully-convolutional inference with linear blending.

    A 1024x2048 frame in fp32 needs several GB of activations.  We process
    512x512 tiles with a 64 px overlap and blend them linearly, which keeps
    peak VRAM well under the 4 GB budget while matching full-frame output
    closely.
    """
    _, _, h, w = lq.shape
    if h <= tile and w <= tile:
        with torch.amp.autocast("cuda", enabled=lq.is_cuda):
            return model(lq).float()

    out = torch.zeros_like(lq, dtype=torch.float32)
    weight = torch.zeros((1, 1, h, w), device=lq.device, dtype=torch.float32)
    step = tile - overlap
    ys = list(range(0, max(h - tile, 0) + 1, step))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    xs = list(range(0, max(w - tile, 0) + 1, step))
    if xs[-1] + tile < w:
        xs.append(w - tile)

    for y in ys:
        for x in xs:
            y2, x2 = min(y + tile, h), min(x + tile, w)
            y, x = max(0, y2 - tile), max(0, x2 - tile)
            patch = lq[:, :, y:y2, x:x2]
            with torch.amp.autocast("cuda", enabled=lq.is_cuda):
                pred = model(patch).float()

            wy = torch.ones(y2 - y, device=lq.device, dtype=torch.float32)
            wx = torch.ones(x2 - x, device=lq.device, dtype=torch.float32)
            if y > 0:
                wy[:overlap] = torch.linspace(0, 1, overlap, device=lq.device)
            if y2 < h:
                wy[-overlap:] = torch.linspace(1, 0, overlap, device=lq.device)
            if x > 0:
                wx[:overlap] = torch.linspace(0, 1, overlap, device=lq.device)
            if x2 < w:
                wx[-overlap:] = torch.linspace(1, 0, overlap, device=lq.device)
            blend = (wy[:, None] * wx[None, :])[None, None]

            out[:, :, y:y2, x:x2] += pred * blend
            weight[:, :, y:y2, x:x2] += blend
    return out / weight.clamp_min(1e-6)


@torch.no_grad()
def evaluate(config: dict, name: str, split: str = "test", tta: bool = False,
             perceptual_size: int = 1024, save_preds: bool = False,
             checkpoint: str | None = None, tile: int = 768) -> dict:
    device = get_device()
    model = build_model(config["model"]).to(device).eval()
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"Checkpoint: {checkpoint}")
    print(f"Device: {describe_device()} | params {count_parameters_m(model):.3f} M")

    lpips_metric = LPIPSMetric(device)
    dists_metric = DISTSMetric(device)

    out_root = Path(config["output"]["root"]) / "eval" / name
    out_root.mkdir(parents=True, exist_ok=True)
    pred_dir = out_root / "predictions"
    if save_preds:
        pred_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    all_rows: list[dict] = []

    for ds_name in config["data"].get("eval_datasets", ("poled", "toled")):
        ds = UDCMatDataset(
            data_dir=config["data"]["mat_root"],
            split=split,
            datasets=(ds_name,),
            crop_size=None,
        )
        rows = []
        t_start = time.perf_counter()
        for idx in range(len(ds)):
            batch = ds[idx]
            lq = batch["lq"].unsqueeze(0).to(device)
            gt = batch["gt"].unsqueeze(0).to(device)

            t0 = time.perf_counter()
            pred = tiled_predict(model, lq, tile=tile).clamp(0, 1)
            if tta:
                pred_h = torch.flip(tiled_predict(model, torch.flip(lq, dims=[3]), tile=tile), dims=[3]).clamp(0, 1)
                pred_v = torch.flip(tiled_predict(model, torch.flip(lq, dims=[2]), tile=tile), dims=[2]).clamp(0, 1)
                pred = ((pred + pred_h + pred_v) / 3.0).clamp(0, 1)
            infer_s = time.perf_counter() - t0
            if device.type == "cuda":
                torch.cuda.empty_cache()

            p_psnr = psnr(pred, gt).item()
            p_ssim = ssim(pred, gt).item()
            pp, gg = _perceptual_pair(pred, gt, perceptual_size)
            try:
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    p_lpips = lpips_metric(pp, gg).mean().item()
                    p_dists = dists_metric(pp, gg).mean().item()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                pp, gg = pp.cpu(), gg.cpu()
                lpips_metric.model.cpu(); dists_metric.model.cpu()
                p_lpips = lpips_metric(pp, gg).mean().item()
                p_dists = dists_metric(pp, gg).mean().item()
                lpips_metric.model.to(device); dists_metric.model.to(device)

            row = {
                "dataset": ds_name, "name": batch["name"], "PSNR": p_psnr, "SSIM": p_ssim,
                "LPIPS": p_lpips, "DISTS": p_dists, "infer_s": infer_s,
            }
            rows.append(row)
            all_rows.append(row)
            print(f"  [{ds_name} {idx + 1:02d}/{len(ds)}] PSNR={p_psnr:.4f} SSIM={p_ssim:.4f} "
                  f"LPIPS={p_lpips:.4f} DISTS={p_dists:.4f} ({infer_s:.2f}s)", flush=True)

            if save_preds:
                from PIL import Image
                Image.fromarray(tensor_to_uint8(pred)).save(pred_dir / f"{batch['name']}.png")

        elapsed = time.perf_counter() - t_start
        mean = {
            "images": len(rows),
            "PSNR": float(np.mean([r["PSNR"] for r in rows])),
            "SSIM": float(np.mean([r["SSIM"] for r in rows])),
            "LPIPS": float(np.mean([r["LPIPS"] for r in rows])),
            "DISTS": float(np.mean([r["DISTS"] for r in rows])),
            "seconds_per_image": elapsed / len(rows),
        }
        summary[ds_name] = mean
        save_json(mean, out_root / f"{ds_name}_{split}_metrics.json")

    # Combined + per-image CSV
    combined = {
        "images": len(all_rows),
        "PSNR": float(np.mean([r["PSNR"] for r in all_rows])),
        "SSIM": float(np.mean([r["SSIM"] for r in all_rows])),
        "LPIPS": float(np.mean([r["LPIPS"] for r in all_rows])),
        "DISTS": float(np.mean([r["DISTS"] for r in all_rows])),
        "seconds_per_image": float(np.mean([r["infer_s"] for r in all_rows])),
    }
    summary["combined"] = combined
    summary["parameters_M"] = count_parameters_m(model)
    summary["fusion"] = config["model"]["fusion"]
    summary["tta"] = tta
    summary["perceptual_size"] = perceptual_size
    save_json(summary, out_root / f"{split}_summary.json")

    with open(out_root / f"{split}_per_image.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n=== Summary ===")
    for key, m in summary.items():
        if isinstance(m, dict):
            print(f"{key:10s} PSNR={m['PSNR']:.4f} SSIM={m['SSIM']:.4f} "
                  f"LPIPS={m['LPIPS']:.4f} DISTS={m['DISTS']:.4f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--perceptual-size", type=int, default=1024)
    parser.add_argument("--tile", type=int, default=768)
    parser.add_argument("--save-preds", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "config" in state:
        config = state["config"]

    evaluate(config, args.name, split=args.split, tta=args.tta,
             perceptual_size=args.perceptual_size, save_preds=args.save_preds,
             checkpoint=args.checkpoint, tile=args.tile)


if __name__ == "__main__":
    main()
