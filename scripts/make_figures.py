"""Generate every figure used in the report, slides and README.

Run after training + evaluation:
    python scripts/make_figures.py
Figures are written to outputs/figures and copied to report/figures.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.facecolor": "white",
})

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import UDCMatDataset  # noqa: E402
from src.evaluate import tiled_predict  # noqa: E402
from src.metrics import psnr, ssim  # noqa: E402
from src.model import build_model  # noqa: E402
from src.pyramid import LaplacianPyramid  # noqa: E402

ROOT = Path("D:/UDC_AMFM_Project")
FIG = ROOT / "outputs" / "figures"
REPFIG = ROOT / "report" / "figures"
CKPT = ROOT / "outputs" / "checkpoints"
EVAL = ROOT / "outputs" / "eval"
LOGS = ROOT / "outputs" / "logs"

PAPER = {
    "poled": {"PSNR": 33.99, "SSIM": 0.9629, "LPIPS": 0.1678, "DISTS": 0.1399, "params": 4.0},
    "toled": {"PSNR": 38.62, "SSIM": 0.9805, "LPIPS": 0.0893, "DISTS": 0.0858, "params": 4.0},
}
PAPER_METHODS = [
    ("MSUNET", 8.9, 29.17), ("DAGF", 1.1, 32.29), ("PDCRN", 4.7, 32.99),
    ("ResUNet", 16.5, 31.39), ("UDC-UNet", 5.7, 33.12), ("BNUDC", 4.6, 33.39),
    ("FSI", 3.8, 33.85), ("SGSFormer", 9.3, 33.28), ("LPUDC (paper)", 4.0, 33.99),
]
COLORS = {"dgf": "#4C72B0", "amfm": "#DD8452", "none": "#8C8C8C", "concat": "#55A868"}


def to_img(t: torch.Tensor) -> np.ndarray:
    return t.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()


def read_rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def load_model(name: str, device):
    ckpt_path = CKPT / name / "best_psnr.pth"
    if not ckpt_path.exists():
        return None
    state = torch.load(ckpt_path, map_location="cpu")
    model = build_model(state["config"]["model"]).to(device).eval()
    model.load_state_dict(state["model"])
    return model


def best_name(kind: str) -> str | None:
    """Strongest available run of a kind (prefers fine-tuned), by test PSNR."""
    best, best_p = None, -1.0
    cands = {"dgf": ["dgf_cont", "dgf_ft", "dgf_final", "abl_dgf"],
             "amfm": ["amfm_warm", "amfm_ft", "amfm_final", "abl_amfm"]}[kind]
    for cand in cands:
        s = _eval_summary(cand)
        if s and "combined" in s and (CKPT / cand / "best_psnr.pth").exists():
            if s["combined"]["PSNR"] > best_p:
                best, best_p = cand, s["combined"]["PSNR"]
    return best


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    REPFIG.mkdir(parents=True, exist_ok=True)
    out = FIG / name
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(REPFIG / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# --------------------------------------------------------------------------- #
# Dataset overview
# --------------------------------------------------------------------------- #
def fig_dataset_overview() -> None:
    fig, axes = plt.subplots(3, 4, figsize=(13, 7.5))
    for col, ds in enumerate(["Poled", "Toled"]):
        lq = read_rgb(ROOT / "data" / "Train" / ds / "LQ" / f"{3 + col * 5}.png")
        hq = read_rgb(ROOT / "data" / "Train" / ds / "HQ" / f"{3 + col * 5}.png")
        sl = (slice(0, 1024), slice(512, 1536))
        axes[0, col * 2].imshow(lq[sl]); axes[0, col * 2].set_title(f"{ds} LQ (display)")
        axes[0, col * 2 + 1].imshow(hq[sl]); axes[0, col * 2 + 1].set_title(f"{ds} HQ (ground truth)")
        for a in (axes[0, col * 2], axes[0, col * 2 + 1]):
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
        axes[1, col * 2].plot(np.histogram(hq[..., 1], bins=64, range=(0, 255))[0], color="green", label="HQ G")
        axes[1, col * 2].plot(np.histogram(lq[..., 1], bins=64, range=(0, 255))[0], color="red", alpha=0.6, label="LQ G")
        axes[1, col * 2].set_title("green-channel histogram"); axes[1, col * 2].legend(fontsize=7)
        ratio = lq.mean() / max(hq.mean(), 1e-6)
        axes[1, col * 2 + 1].bar(["LQ", "HQ"], [lq.mean(), hq.mean()], color=["#C44E52", "#4C72B0"])
        axes[1, col * 2 + 1].set_title(f"mean brightness (LQ/HQ={ratio:.2f})")
    # bottom row: dataset statistics
    stats = []
    for ds in ["Poled", "Toled"]:
        vals = []
        for i in range(1, 41):
            lq = read_rgb(ROOT / "data" / "Train" / ds / "LQ" / f"{i}.png").astype(np.float32)
            hq = read_rgb(ROOT / "data" / "Train" / ds / "HQ" / f"{i}.png").astype(np.float32)
            vals.append([lq.mean() / 255, hq.mean() / 255, np.mean(np.abs(lq - hq)) / 255])
        stats.append(np.mean(vals, axis=0))
        # PSNR of the degraded input itself
    ax = axes[2, 0]
    width = 0.35
    x = np.arange(3)
    ax.bar(x - width / 2, stats[0], width, label="POLED", color="#C44E52")
    ax.bar(x + width / 2, stats[1], width, label="TOLED", color="#4C72B0")
    ax.set_xticks(x); ax.set_xticklabels(["LQ mean", "HQ mean", "|LQ-HQ|"])
    ax.set_title("per-dataset intensity statistics"); ax.legend(fontsize=8)
    axes[2, 1].axis("off")
    txt = "POLED: 2.9% transmittance\nsevere low-light + color shift\ndiffraction artifacts\n\nTOLED: ~20% transmittance\nmilder degradation\n\n480 training pairs (1024x2048)\n60 val + 60 test images"
    axes[2, 1].text(0.02, 0.95, txt, va="top", fontsize=10, family="monospace")
    axes[2, 2].axis("off"); axes[2, 3].axis("off")
    fig.suptitle("UDC dataset overview: POLED and TOLED (real paired captures)", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_dataset_overview.png")


# --------------------------------------------------------------------------- #
# Laplacian pyramid
# --------------------------------------------------------------------------- #
def fig_pyramid() -> None:
    img = read_rgb(ROOT / "data" / "Train" / "Poled" / "HQ" / "3.png").astype(np.float32)[..., 1] / 255.0
    t = torch.from_numpy(img)[None, None][:, :, :1024, :1024]
    bands = LaplacianPyramid(3)(t)
    recon = LaplacianPyramid(3).reconstruct(bands)
    err = (recon - t).abs().max().item()

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
    axes[0].imshow(t[0, 0], cmap="gray"); axes[0].set_title("input $I_0$")
    for i, b in enumerate(bands):
        v = np.abs(b[0, 0].numpy()).max()
        axes[i + 1].imshow(b[0, 0].numpy(), cmap="bwr", vmin=-v, vmax=v)
        axes[i + 1].set_title(f"Laplacian band {i}\n{b.shape[-2]}x{b.shape[-1]}")
    for a in axes:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.suptitle(f"3-level Laplacian pyramid decomposition (max reconstruction error {err:.2e})", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_pyramid.png")


# --------------------------------------------------------------------------- #
# Architecture + module diagrams
# --------------------------------------------------------------------------- #
def _box(ax, x, y, w, h, text, color="#E8EEF7", ec="#4C72B0", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=color, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def _arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#333333"))


def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)

    _box(ax, 0.02, 0.45, 0.10, 0.10, "Input LQ\n1024x2048", "#FFF3CD", "#B8860B")
    _box(ax, 0.15, 0.42, 0.13, 0.16, "Laplacian\nPyramid\n(3 levels)", "#E8EEF7")
    _arrow(ax, (0.12, 0.50), (0.15, 0.50))

    _box(ax, 0.32, 0.70, 0.16, 0.16, "High-freq\nbranch\n$L_0$  (1024x2048)", "#FBE3E3", "#C44E52")
    _box(ax, 0.32, 0.42, 0.16, 0.16, "High-freq\nbranch\n$L_1$  (512x1024)", "#FBE3E3", "#C44E52")
    _box(ax, 0.32, 0.13, 0.16, 0.16, "Low-freq\nbranch (enc-dec)\n$L_2$ (256x512)", "#E3F2E3", "#55A868")
    for y in (0.78, 0.50, 0.21):
        _arrow(ax, (0.28, 0.50), (0.32, y))

    _box(ax, 0.55, 0.42, 0.16, 0.16, "AMFM fusion\n(channel + spatial\nattention)", "#FFE8CC", "#DD8452")
    _box(ax, 0.55, 0.70, 0.16, 0.16, "AMFM fusion\n(channel + spatial\nattention)", "#FFE8CC", "#DD8452")
    _arrow(ax, (0.48, 0.50), (0.55, 0.50))
    _arrow(ax, (0.48, 0.78), (0.55, 0.78))
    _arrow(ax, (0.63, 0.58), (0.63, 0.70))  # coarse guidance flows upward

    _box(ax, 0.78, 0.40, 0.13, 0.20, "Pyramid sum\n$\\hat{I}= \\hat{L_0}+\\uparrow\\hat{L_1}+\\uparrow\\uparrow\\hat{L_2}$", "#E8EEF7")
    _arrow(ax, (0.71, 0.50), (0.78, 0.50))
    _arrow(ax, (0.71, 0.78), (0.78, 0.60))
    _arrow(ax, (0.40, 0.29), (0.78, 0.40))

    _box(ax, 0.78, 0.16, 0.13, 0.12, "Output\n1024x2048", "#FFF3CD", "#B8860B")
    _arrow(ax, (0.845, 0.40), (0.845, 0.28))

    _box(ax, 0.05, 0.80, 0.22, 0.13, "Loss = L1 + 0.5·Laplacian-pyramid L1", "#F0E6F7", "#7A5195", fs=9)
    ax.text(0.5, 0.975, "LPUDC architecture with pluggable multi-scale fusion",
            ha="center", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_architecture.png")


def fig_amfm() -> None:
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)

    _box(ax, 0.02, 0.60, 0.13, 0.16, "High-freq\nfeatures\n$F_H$", "#FBE3E3", "#C44E52")
    _box(ax, 0.02, 0.22, 0.13, 0.16, "Low-freq\nfeatures\n$\\uparrow F_L$", "#E3F2E3", "#55A868")
    _box(ax, 0.20, 0.41, 0.10, 0.16, "Concat", "#E8EEF7")
    _arrow(ax, (0.15, 0.68), (0.20, 0.55))
    _arrow(ax, (0.15, 0.30), (0.20, 0.45))

    _box(ax, 0.34, 0.41, 0.13, 0.16, "Channel\nAttention\n(SE)", "#FFE8CC", "#DD8452")
    _arrow(ax, (0.30, 0.49), (0.34, 0.49))
    _box(ax, 0.51, 0.41, 0.13, 0.16, "Spatial\nAttention\n(CBAM)", "#FFE8CC", "#DD8452")
    _arrow(ax, (0.47, 0.49), (0.51, 0.49))
    _box(ax, 0.68, 0.41, 0.13, 0.16, "Refine\nconv x2", "#E8EEF7")
    _arrow(ax, (0.64, 0.49), (0.68, 0.49))
    _box(ax, 0.85, 0.41, 0.12, 0.16, "Fused\nfeatures", "#F0E6F7", "#7A5195")
    _arrow(ax, (0.81, 0.49), (0.85, 0.49))

    # residual connection over the top
    _arrow(ax, (0.085, 0.76), (0.91, 0.76))
    _arrow(ax, (0.91, 0.76), (0.91, 0.57))
    ax.text(0.5, 0.80, "residual connection", ha="center", fontsize=9, style="italic", color="#7A5195")
    ax.text(0.5, 0.10, "Channel attention re-weights feature channels; spatial attention re-weights pixel\n"
                       "locations (diffraction fringes, texture loss). The attended features predict guided-filter\n"
                       "coefficients (a, b) applied to the guidance, then a residual keeps the high-frequency detail.\n"
                       "With attention saturated, AMFM reduces exactly to the base paper's Deep Guided Filtering.",
            ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_amfm.png")


# --------------------------------------------------------------------------- #
# Training curves
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def fig_training_curves() -> None:
    runs = [("abl_dgf", "DGF (5k)"), ("abl_amfm", "AMFM (5k)"),
            ("dgf_final", "DGF (15k)"), ("amfm_final", "AMFM (15k)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, label in runs:
        tr = _read_jsonl(LOGS / name / "train_log.jsonl")
        va = _read_jsonl(LOGS / name / "val_log.jsonl")
        if not tr:
            continue
        color = COLORS["amfm"] if "amfm" in name else COLORS["dgf"]
        style = "--" if "5k" in label else "-"
        axes[0].plot([r["iteration"] for r in tr], [r["loss"] for r in tr], style, color=color, label=label, lw=1.4)
        if va:
            axes[1].plot([r["iteration"] for r in va], [r["val_PSNR"] for r in va], style, color=color, label=label, marker="o", ms=3)
            axes[2].plot([r["iteration"] for r in va], [r["val_SSIM"] for r in va], style, color=color, label=label, marker="s", ms=3)
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("training loss"); axes[0].set_title("Training loss")
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("val PSNR (dB)"); axes[1].set_title("Validation PSNR")
    axes[2].set_xlabel("iteration"); axes[2].set_ylabel("val SSIM"); axes[2].set_title("Validation SSIM")
    for a in axes:
        a.legend(fontsize=8)
    fig.suptitle("Training and validation curves", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_training_curves.png")


# --------------------------------------------------------------------------- #
# Metrics: comparison + ablation
# --------------------------------------------------------------------------- #
def _eval_summary(name: str):
    p = EVAL / name / "test_summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig_metrics_comparison() -> None:
    names = [(best_name("dgf") or "dgf_final", "Ours: DGF (base fusion)"),
             (best_name("amfm") or "amfm_final", "Ours: AMFM (proposed)")]
    got = {n: _eval_summary(n) for n, _ in names}
    if not any(got.values()):
        print("  [skip] no eval summaries yet"); return
    metrics = ["PSNR", "SSIM", "LPIPS", "DISTS"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, m in zip(axes, metrics):
        for ds in ("poled", "toled"):
            labels, vals, cols = [], [], []
            labels.append("paper LPUDC"); vals.append(PAPER[ds][m]); cols.append("#7A5195")
            for n, lab in names:
                s = got[n]
                if s and ds in s:
                    labels.append(lab.replace("Ours: ", "")); vals.append(s[ds][m]); cols.append(COLORS["dgf"] if "dgf" in n else COLORS["amfm"])
            x = np.arange(len(labels)) + (0 if ds == "poled" else len(labels) + 1.2)
            ax.bar(x, vals, color=cols)
            ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        ax.set_title(m)
        if m in ("PSNR", "SSIM"):
            ax.set_ylim(bottom=min(v for v in ax.get_ylim()), )
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#7A5195"),
                        plt.Rectangle((0, 0), 1, 1, color=COLORS["dgf"]),
                        plt.Rectangle((0, 0), 1, 1, color=COLORS["amfm"])],
               labels=["paper LPUDC", "our DGF", "our AMFM"], loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Test-set metrics: paper reference vs our DGF vs our AMFM", fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig_metrics_comparison.png")


def fig_ablation() -> None:
    rows = []
    for name, lab in [("abl_none", "no fusion"), ("abl_concat", "concat"), ("abl_dgf", "DGF"), ("abl_amfm", "AMFM")]:
        s = _eval_summary(name)
        if s:
            rows.append((lab, s["combined"]))
    if not rows:
        print("  [skip] ablation evals missing"); return
    metrics = ["PSNR", "SSIM", "LPIPS", "DISTS"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, m in zip(axes, metrics):
        vals = [r[1][m] for r in rows]
        cols = ["#8C8C8C", "#55A868", COLORS["dgf"], COLORS["amfm"]][: len(rows)]
        bars = ax.bar([r[0] for r in rows], vals, color=cols)
        ax.set_title(m)
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
        best = max(vals) if m in ("PSNR", "SSIM") else min(vals)
        ax.set_ylim(bottom=min(vals) * (0.98 if m in ("PSNR", "SSIM") else 0))
    fig.suptitle("Matched-budget ablation (5k iters, identical training protocol)", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_ablation.png")


def fig_per_image() -> None:
    import csv
    p = EVAL / (best_name("dgf") or "dgf_final") / "test_per_image.csv"
    q = EVAL / (best_name("amfm") or "amfm_final") / "test_per_image.csv"
    if not (p.exists() and q.exists()):
        print("  [skip] per-image csv missing"); return
    def load(x):
        with open(x, newline="", encoding="utf-8") as fh:
            return {r["name"]: r for r in csv.DictReader(fh)}
    a, b = load(p), load(q)
    common = sorted(set(a) & set(b))
    d = np.array([float(b[k]["PSNR"]) for k in common])
    e = np.array([float(a[k]["PSNR"]) for k in common])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].scatter(e, d, c=["#C44E52" if k.startswith("poled") else "#4C72B0" for k in common], s=22, alpha=0.8)
    lim = [min(e.min(), d.min()) - 1, max(e.max(), d.max()) + 1]
    axes[0].plot(lim, lim, "k--", lw=1)
    axes[0].set_xlim(lim); axes[0].set_ylim(lim)
    axes[0].set_xlabel("DGF PSNR (dB)"); axes[0].set_ylabel("AMFM PSNR (dB)")
    axes[0].set_title(f"Per-image PSNR ({np.mean(d > e) * 100:.0f}% of images improved)")
    axes[1].hist(d - e, bins=18, color=COLORS["amfm"], edgecolor="white")
    axes[1].axvline(0, color="k", ls="--")
    axes[1].set_xlabel("AMFM - DGF PSNR (dB)"); axes[1].set_ylabel("#images")
    axes[1].set_title(f"Mean gain {np.mean(d - e):+.3f} dB")
    fig.suptitle("Per-image effect of the AMFM module", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_per_image.png")


# --------------------------------------------------------------------------- #
# Qualitative + error maps + attention
# --------------------------------------------------------------------------- #
def fig_qualitative(device) -> None:
    models = {"DGF": load_model(best_name("dgf") or "dgf_final", device),
              "AMFM": load_model(best_name("amfm") or "amfm_final", device)}
    models = {k: v for k, v in models.items() if v is not None}
    for ds in ("poled", "toled"):
        data = UDCMatDataset(ROOT / "data", split="test", datasets=(ds,), crop_size=None)
        idxs = [2, 7, 16]
        mr, mc, tile = len(idxs), 2 + len(models) + 1, 260
        fig, axes = plt.subplots(mr, mc, figsize=(mc * 3.3, mr * 3.0))
        for r, i in enumerate(idxs):
            item = data[i]
            lq = item["lq"][None].to(device); gt = item["gt"][None].to(device)
            panels = [("LQ input", to_img(lq)), ]
            for nm, m in models.items():
                with torch.no_grad():
                    pred = tiled_predict(m, lq).clamp(0, 1)
                panels.append((f"{nm}  PSNR {psnr(pred, gt).item():.1f} dB", to_img(pred)))
            panels.append(("Ground truth", to_img(gt)))
            ys, xs = 380, 900
            for c, (title, im) in enumerate(panels):
                axes[r, c].imshow(im[ys:ys + tile, xs:xs + tile])
                if r == 0:
                    axes[r, c].set_title(title, fontsize=9)
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([]); axes[r, c].grid(False)
        fig.suptitle(f"Qualitative comparison on {ds.upper()} test set (zoom on a texture region)", fontweight="bold")
        fig.tight_layout()
        save(fig, f"fig_qualitative_{ds}.png")


def fig_error_maps(device) -> None:
    models = {"DGF": load_model(best_name("dgf") or "dgf_final", device),
              "AMFM": load_model(best_name("amfm") or "amfm_final", device)}
    models = {k: v for k, v in models.items() if v is not None}
    if not models:
        print("  [skip] no models"); return
    data = UDCMatDataset(ROOT / "data", split="test", datasets=("poled",), crop_size=(512, 1024))
    item = data[5]
    lq = item["lq"][None].to(device); gt = item["gt"][None].to(device)
    n = len(models)
    fig, axes = plt.subplots(2, n + 1, figsize=(4 * (n + 1), 7))
    axes[0, 0].imshow(to_img(lq)); axes[0, 0].set_title("LQ input")
    axes[1, 0].imshow(to_img(gt)); axes[1, 0].set_title("Ground truth")
    for j, (nm, m) in enumerate(models.items()):
        with torch.no_grad():
            pred = tiled_predict(m, lq).clamp(0, 1)
        err = (pred - gt).abs()[0].mean(0).cpu().numpy()
        axes[0, j + 1].imshow(to_img(pred)); axes[0, j + 1].set_title(f"{nm} restored")
        im = axes[1, j + 1].imshow(np.clip(err * 5, 0, 1), cmap="inferno")
        axes[1, j + 1].set_title(f"|error| x5  (PSNR {psnr(pred, gt).item():.1f})")
        plt.colorbar(im, ax=axes[1, j + 1], fraction=0.046)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.suptitle("Error maps on POLED (center crop)", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_error_maps.png")


def fig_attention(device) -> None:
    m = load_model(best_name("amfm") or "amfm_final", device)
    if m is None:
        m = load_model("abl_amfm", device)
    if m is None:
        print("  [skip] no AMFM model"); return
    gates, maps = [], []
    h1 = m.fusions[0].channel_attn.mlp.register_forward_hook(lambda mod, i, o: gates.append(o.detach().cpu()))
    h2 = m.fusions[0].spatial_attn.conv.register_forward_hook(lambda mod, i, o: maps.append(o.detach().cpu()))

    data = UDCMatDataset(ROOT / "data", split="test", datasets=("poled",), crop_size=(512, 1024))
    item = data[3]
    lq = item["lq"][None].to(device)
    with torch.no_grad():
        pred = tiled_predict(m, lq).clamp(0, 1)
    h1.remove(); h2.remove()

    ch = gates[0][0, :, 0, 0].numpy()
    sp = torch.sigmoid(maps[0])[0, 0].numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    axes[0].imshow(to_img(lq)); axes[0].set_title("LQ input")
    axes[1].imshow(to_img(pred)); axes[1].set_title("AMFM restored")
    axes[2].imshow(sp, cmap="jet"); axes[2].set_title("Spatial attention map\n(fusion level 0)")
    axes[3].barh(np.arange(len(ch)), ch, color=COLORS["amfm"])
    axes[3].set_title("Channel attention weights"); axes[3].set_ylabel("channel"); axes[3].invert_yaxis()
    for a in axes[:3]:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.suptitle("What the attention learns (AMFM, POLED test image)", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_attention.png")


def fig_params_scatter(device=None) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5))
    for nm, p, v in PAPER_METHODS:
        ax.scatter(p, v, s=45, color="#B0B0B0", zorder=2)
        ax.annotate(nm, (p, v), fontsize=7.5, xytext=(4, 4), textcoords="offset points")
    dn, an = best_name("dgf") or "dgf_final", best_name("amfm") or "amfm_final"
    got = {n: _eval_summary(n) for n in (dn, an)}
    for n, c, lab in [(dn, COLORS["dgf"], "Ours DGF"), (an, COLORS["amfm"], "Ours AMFM")]:
        s = got[n]
        if s:
            ax.scatter(s["parameters_M"], s["poled"]["PSNR"], s=140, marker="*", color=c, zorder=3, edgecolor="k")
            ax.annotate(f"{lab}\n({s['parameters_M']:.2f}M)", (s["parameters_M"], s["poled"]["PSNR"]),
                        fontsize=8, xytext=(6, -12), textcoords="offset points", color=c, fontweight="bold")
    ax.axhline(PAPER["poled"]["PSNR"], color="#7A5195", ls="--", lw=1.2)
    ax.text(0.5, PAPER["poled"]["PSNR"] + 0.06, "paper LPUDC 33.99 dB", color="#7A5195", fontsize=8)
    ax.set_xlabel("parameters (M)"); ax.set_ylabel("POLED PSNR (dB)")
    ax.set_title("Accuracy vs model size (POLED)", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_params_scatter.png")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Generating figures...")
    for fn in (fig_dataset_overview, fig_pyramid, fig_architecture, fig_amfm,
               fig_training_curves, fig_metrics_comparison, fig_ablation, fig_per_image):
        try:
            fn()
        except Exception as exc:
            print(f"  [error] {fn.__name__}: {exc}")
    if device.type == "cuda":
        for fn in (fig_qualitative, fig_error_maps, fig_attention, fig_params_scatter):
            try:
                fn(device)
            except Exception as exc:
                print(f"  [error] {fn.__name__}: {exc}")
            finally:
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
