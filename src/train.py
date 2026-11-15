"""Train LPUDC with a selectable fusion module.

Usage
-----
python -m src.train --config configs/amfm.yaml --name amfm
python -m src.train --config configs/dgf.yaml  --name dgf
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import argparse
import copy
import csv
import time
from pathlib import Path

import torch
import yaml

from .data import build_train_loader, build_val_loader
from .losses import LPUDCLoss
from .metrics import psnr, ssim
from .model import build_model
from .utils import (
    AverageMeter,
    append_jsonl,
    cosine_lr,
    count_parameters_m,
    describe_device,
    get_device,
    human_time,
    save_json,
    seed_everything,
)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def apply_overrides(config: dict, args) -> dict:
    if args.fusion:
        config["model"]["fusion"] = args.fusion
    if args.width:
        config["model"]["width"] = args.width
    if args.iterations:
        config["train"]["iterations"] = args.iterations
    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
    if args.lr:
        config["train"]["lr"] = args.lr
    if args.crop:
        config["data"]["crop"] = args.crop
    if args.num_workers is not None:
        config["train"]["num_workers"] = args.num_workers
    return config


@torch.no_grad()
def validate(model, loader, device) -> dict[str, float]:
    model.eval()
    psnrs, ssims = [], []
    for batch in loader:
        lq = batch["lq"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            pred = model(lq)
        pred = pred.float().clamp(0, 1)
        psnrs.append(psnr(pred, gt).item())
        ssims.append(ssim(pred, gt).item())
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "PSNR": sum(psnrs) / max(len(psnrs), 1),
        "SSIM": sum(ssims) / max(len(ssims), 1),
    }


def train(config: dict, name: str, init_weights: str | None = None) -> dict:
    seed_everything(config.get("seed", 42))
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    out_root = Path(config["output"]["root"])
    ckpt_dir = out_root / "checkpoints" / name
    log_dir = out_root / "logs" / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"Experiment : {name}")
    print(f"Fusion     : {config['model']['fusion']}")
    print(f"Device     : {describe_device()}")
    print("=" * 78)

    model = build_model(config["model"]).to(device)
    print(f"Parameters : {count_parameters_m(model):.3f} M")

    if init_weights:
        state = torch.load(init_weights, map_location="cpu")
        state = state.get("model", state) if isinstance(state, dict) else state
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Init from  : {init_weights} (missing={len(missing)}, unexpected={len(unexpected)})")

    train_loader, train_ds = build_train_loader(config, num_workers=config["train"].get("num_workers", 6))
    val_loader, val_ds = build_val_loader(config, split="val", full=False)
    print(f"Train pairs: {len(train_ds)} | Val images: {len(val_ds)}")

    criterion = LPUDCLoss(
        lambda_pyramid=config["train"].get("lambda_pyramid", 0.5),
        levels=config["model"].get("levels", 3),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        betas=(0.9, 0.999),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Exponential moving average of the weights (cheap, reliable accuracy boost).
    use_ema = bool(config["train"].get("ema", True))
    ema_decay = float(config["train"].get("ema_decay", 0.999))
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    _ema_pairs = [
        (e, m)
        for e, m in zip(ema_model.state_dict().values(), model.state_dict().values())
    ]

    total_iters = int(config["train"]["iterations"])
    val_every = int(config["train"].get("val_every", 2000))
    log_every = int(config["train"].get("log_every", 50))
    grad_clip = float(config["train"].get("grad_clip", 1.0))
    base_lr = float(config["train"]["lr"])
    min_lr = float(config["train"].get("min_lr", 1e-6))
    warmup = int(config["train"].get("warmup", 500))

    metrics_path = log_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["iteration", "total_loss", "content_loss", "pyramid_loss", "lr", "val_psnr", "val_ssim", "elapsed_s"])
    train_log_path = log_dir / "train_log.jsonl"
    val_log_path = log_dir / "val_log.jsonl"
    for p in (train_log_path, val_log_path):
        if p.exists():
            p.unlink()

    best = {"psnr": 0.0, "ssim": 0.0, "psnr_iter": 0, "ssim_iter": 0}
    meter = AverageMeter()
    t0 = time.perf_counter()
    model.train()
    step = 0
    epoch = 0

    while step < total_iters:
        epoch += 1
        for batch in train_loader:
            if step >= total_iters:
                break
            lr = cosine_lr(step, total_iters, base_lr, min_lr, warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr

            lq = batch["lq"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(lq)
                loss, parts = criterion(pred, gt)

            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            if use_ema and (step % int(config["train"].get("ema_every", 2)) == 0):
                with torch.no_grad():
                    for e, m in _ema_pairs:
                        if e.dtype.is_floating_point:
                            e.mul_(ema_decay).add_(m.detach(), alpha=1.0 - ema_decay)
                        else:
                            e.copy_(m)

            meter.update(loss.item())
            elapsed = time.perf_counter() - t0

            if step % log_every == 0 or step == total_iters - 1:
                row = {
                    "iteration": step,
                    "loss": loss.item(),
                    "content": parts["content"].item(),
                    "pyramid": parts["pyramid"].item(),
                    "lr": lr,
                    "elapsed_s": elapsed,
                }
                append_jsonl(row, train_log_path)
                print(
                    f"[{step:6d}/{total_iters}] loss={meter.avg:.5f} "
                    f"(c={parts['content'].item():.4f}, p={parts['pyramid'].item():.4f}) "
                    f"lr={lr:.2e} {elapsed:7.1f}s",
                    flush=True,
                )

            if step > 0 and (step % val_every == 0 or step == total_iters - 1):
                val = validate(model, val_loader, device)
                ema_val = validate(ema_model, val_loader, device) if use_ema else None
                use_ema_ckpt = bool(ema_val and ema_val["PSNR"] >= val["PSNR"])
                chosen = ema_val if use_ema_ckpt else val
                tag = "ema" if use_ema_ckpt else "raw"
                append_jsonl({"iteration": step, "weights": tag,
                              **{f"val_{k}": v for k, v in chosen.items()},
                              "val_ema_PSNR": (ema_val or {}).get("PSNR")}, val_log_path)
                print(f"    val  PSNR={val['PSNR']:.4f}  SSIM={val['SSIM']:.4f}"
                      + (f"  | ema PSNR={ema_val['PSNR']:.4f} SSIM={ema_val['SSIM']:.4f} [{tag}]"
                         if ema_val else ""), flush=True)
                with open(metrics_path, "a", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerow(
                        [step, meter.avg, parts["content"].item(), parts["pyramid"].item(),
                         lr, chosen["PSNR"], chosen["SSIM"], elapsed]
                    )
                state = (ema_model if use_ema_ckpt else model).state_dict()
                ckpt = {
                    "model": state,
                    "iteration": step,
                    "config": config,
                    "val": chosen,
                    "weights": tag,
                    "name": name,
                }
                torch.save(ckpt, ckpt_dir / "latest.pth")
                if chosen["PSNR"] > best["psnr"]:
                    best.update(psnr=chosen["PSNR"], psnr_iter=step)
                    torch.save(ckpt, ckpt_dir / "best_psnr.pth")
                if chosen["SSIM"] > best["ssim"]:
                    best.update(ssim=chosen["SSIM"], ssim_iter=step)
                    torch.save(ckpt, ckpt_dir / "best_ssim.pth")

            step += 1

    with open(ckpt_dir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)

    summary = {
        "name": name,
        "fusion": config["model"]["fusion"],
        "parameters_M": count_parameters_m(model),
        "iterations": total_iters,
        "train_seconds": time.perf_counter() - t0,
        "best_val_psnr": best["psnr"],
        "best_val_psnr_iter": best["psnr_iter"],
        "best_val_ssim": best["ssim"],
        "best_val_ssim_iter": best["ssim_iter"],
    }
    save_json(summary, log_dir / "summary.json")
    print(f"Done in {human_time(summary['train_seconds'])} | best val PSNR {best['psnr']:.4f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--init-weights", default=None)
    parser.add_argument("--fusion", default=None, choices=["none", "concat", "dgf", "amfm"])
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--crop", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args)
    train(config, args.name, init_weights=args.init_weights)


if __name__ == "__main__":
    main()
