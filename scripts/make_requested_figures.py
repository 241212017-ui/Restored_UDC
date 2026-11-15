"""Generate the four requested figures:

    fig1_udc_degradation.png   UDC degradation illustration (real + labelled simulation)
    fig2_amfm_architecture.png AMFM module architecture
    fig3_psnr_comparison.png   PSNR comparison vs reported SOTA (POLED + TOLED)
    fig4_metrics.png           Full metric comparison (PSNR / SSIM / LPIPS / DISTS)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams.update({"figure.dpi": 140, "savefig.dpi": 200, "font.size": 10,
                            "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white"})

ROOT = Path("D:/UDC_AMFM_Project")
FIG = ROOT / "outputs" / "figures"
REPFIG = ROOT / "report" / "figures"
EVAL = ROOT / "outputs" / "eval"

C_DGF, C_AMFM, C_PAPER = "#4C72B0", "#DD8452", "#7A5195"
PAPER_POLED = {"PSNR": 33.99, "SSIM": 0.9629, "LPIPS": 0.1678, "DISTS": 0.1399}
PAPER_TOLED = {"PSNR": 38.62, "SSIM": 0.9805, "LPIPS": 0.0893, "DISTS": 0.0858}
SOTA_POLED = [("MSUNET", 29.17), ("DAGF", 32.29), ("PDCRN", 32.99), ("ResUNet", 31.39),
              ("UDC-UNet", 33.12), ("BNUDC", 33.39), ("SGSFormer", 33.28), ("FSI", 33.85),
              ("LPUDC", 33.99)]
SOTA_TOLED = [("MSUNET", 37.40), ("DAGF", 36.49), ("IPIUer", 38.18), ("BAIDU", 38.23),
              ("UDC-UNet", 38.10), ("BNUDC", 38.22), ("SGSFormer", 38.42), ("FSI", 38.60),
              ("LPUDC", 38.62)]


def summary(name):
    p = EVAL / name / "test_summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    REPFIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight")
    fig.savefig(REPFIG / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


# --------------------------------------------------------------------------- #
# Figure 1 — UDC degradation
# --------------------------------------------------------------------------- #
def simulated_lowfreq(img):
    """Brightness attenuation + colour shift (transparent layers)."""
    x = img.astype(np.float64) / 255.0
    x = x * 0.42
    x[..., 0] *= 0.90
    x[..., 2] *= 1.18
    x += np.array([0.010, 0.014, 0.026])
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def diffraction_psf(size=31, spacing=6, lobes=6):
    """Simple multi-lobe PSF that mimics OLED wiring diffraction."""
    k = np.zeros((size, size))
    c = size // 2
    k[c, c] = 1.0
    for r in range(1, lobes + 1):
        d = spacing * r
        amp = 0.30 / r
        for dy, dx in ((0, d), (0, -d), (d, 0), (-d, 0), (d, d), (-d, -d), (d, -d), (-d, d)):
            y, x = c + dy, c + dx
            if 0 <= y < size and 0 <= x < size:
                k[y, x] = amp
    k /= k.sum()
    return k


def simulated_highfreq(img, psf):
    x = img.astype(np.float64) / 255.0
    out = np.stack([cv2.filter2D(x[..., c], -1, psf, borderType=cv2.BORDER_REPLICATE) for c in range(3)], -1)
    # boost contrast slightly + add fine structural noise
    out = np.clip(out, 0, 1)
    rng = np.random.default_rng(0)
    out += rng.normal(0, 0.012, out.shape)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def fig1_udc_degradation():
    gt_path = ROOT / "data" / "Train" / "Poled" / "HQ" / "3.png"
    lq_path = ROOT / "data" / "Train" / "Poled" / "LQ" / "3.png"
    gt = cv2.cvtColor(cv2.imread(str(gt_path)), cv2.COLOR_BGR2RGB)
    lq = cv2.cvtColor(cv2.imread(str(lq_path)), cv2.COLOR_BGR2RGB)
    sl = (slice(120, 760), slice(560, 1480))
    gt, lq = gt[sl], lq[sl]

    low = simulated_lowfreq(gt)
    high = simulated_highfreq(gt, diffraction_psf())

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    panels = [(gt, "(a) Original scene (ground truth)", "#2E7D32"),
              (low, "(b) Low-frequency degradation\nbrightness loss + colour shift (illustrative)", "#B8860B"),
              (high, "(c) High-frequency degradation\ndiffraction ghosts + structural noise (illustrative)", "#C62828"),
              (lq, "(d) Real under-display capture (POLED)", "#1565C0")]
    for ax, (im, title, col) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(title, fontsize=10, color=col)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("Under-display camera imaging: where the degradation comes from", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig1_udc_degradation.png")


# --------------------------------------------------------------------------- #
# Figure 2 — AMFM architecture
# --------------------------------------------------------------------------- #
def _box(ax, x, y, w, h, text, fc="#E8EEF7", ec="#4C72B0", fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, p, q, color="#333333", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=1.4, color=color, linestyle=ls))


def fig2_amfm_architecture():
    fig, ax = plt.subplots(figsize=(14, 6.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)

    _box(ax, 0.015, 0.60, 0.115, 0.16, "High-freq\nfeatures $F_H$", "#FBE3E3", "#C44E52")
    _box(ax, 0.015, 0.22, 0.115, 0.16, "Low-freq\n(guidance) $G$", "#E3F2E3", "#2E7D32")

    _box(ax, 0.175, 0.40, 0.095, 0.16, "Concat\n+ 3x3 conv", "#E8EEF7")
    _arrow(ax, (0.130, 0.68), (0.175, 0.55))
    _arrow(ax, (0.130, 0.30), (0.175, 0.45))

    _box(ax, 0.305, 0.42, 0.115, 0.16, "Channel\nattention\n(SE)", "#FFE8CC", "#DD8452")
    _arrow(ax, (0.270, 0.48), (0.305, 0.48))

    _box(ax, 0.455, 0.42, 0.115, 0.16, "Spatial\nattention\n(CBAM 7x7)", "#FFE8CC", "#DD8452")
    _arrow(ax, (0.420, 0.48), (0.455, 0.48))

    _box(ax, 0.610, 0.42, 0.115, 0.16, "1x1 conv\n-> (a, b)", "#F0E6F7", "#7A5195")
    _arrow(ax, (0.570, 0.48), (0.610, 0.48))

    _box(ax, 0.765, 0.42, 0.115, 0.16, "Guided fusion\n$a \\odot G + b$", "#F0E6F7", "#7A5195")
    _arrow(ax, (0.725, 0.48), (0.765, 0.48))
    _arrow(ax, (0.072, 0.22), (0.072, 0.12))
    _arrow(ax, (0.072, 0.12), (0.822, 0.12))
    _arrow(ax, (0.822, 0.12), (0.822, 0.42), color=C_PAPER)
    ax.text(0.45, 0.075, "guidance $G$", ha="center", fontsize=9, color=C_PAPER)

    _box(ax, 0.765, 0.70, 0.115, 0.16, "Refine\n(3x3 conv)", "#E8EEF7")
    _arrow(ax, (0.822, 0.58), (0.822, 0.70))
    _box(ax, 0.585, 0.70, 0.135, 0.16, "Zero-init 3x3\nconv", "#E8EEF7")
    _arrow(ax, (0.765, 0.78), (0.720, 0.78))
    _box(ax, 0.905, 0.70, 0.085, 0.16, "Output\n$Y$", "#FFF3CD", "#B8860B")
    _arrow(ax, (0.585, 0.78), (0.520, 0.78))
    _arrow(ax, (0.905, 0.78), (0.990, 0.78))

    # residual from F_H
    _arrow(ax, (0.072, 0.76), (0.072, 0.92))
    _arrow(ax, (0.072, 0.92), (0.947, 0.92))
    _arrow(ax, (0.947, 0.92), (0.947, 0.86), color=C_AMFM)
    ax.text(0.5, 0.945, "residual connection  (zero-init  ->  module == DGF at initialisation)",
            ha="center", fontsize=9, color=C_AMFM)

    ax.text(0.5, 0.30,
            "channel attention: which feature channels matter      |      "
            "spatial attention: which pixels matter (diffraction fringes)",
            ha="center", fontsize=9.5, color="#555555")
    ax.text(0.5, 0.20,
            "AMFM = Deep Guided Filtering core + a zero-initialised attention correction branch  "
            "(a strict superset of the base module)",
            ha="center", fontsize=9.5, color="#555555")
    ax.text(0.5, 0.985, "Attention-based Multi-Scale Fusion Module (AMFM)",
            ha="center", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig2_amfm_architecture.png")


# --------------------------------------------------------------------------- #
# Figure 3 — PSNR comparison vs reported SOTA
# --------------------------------------------------------------------------- #
def fig3_psnr_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, sota, ds, ours_ctrl, ours_prop, paper in (
            (axes[0], SOTA_POLED, "POLED", "dgf_cont", "amfm_warm", PAPER_POLED["PSNR"]),
            (axes[1], SOTA_TOLED, "TOLED", "dgf_cont", "amfm_warm", PAPER_TOLED["PSNR"])):
        names = [n for n, _ in sota]
        vals = [v for _, v in sota]
        colors = [C_PAPER if n == "LPUDC" else "#B0B0B0" for n in names]
        sc = summary(ours_ctrl); sp = summary(ours_prop)
        extra_names, extra_vals, extra_cols = [], [], []
        if sc:
            extra_names.append("Ours DGF"); extra_vals.append(sc[ds.lower()]["PSNR"]); extra_cols.append(C_DGF)
        if sp:
            extra_names.append("Ours AMFM"); extra_vals.append(sp[ds.lower()]["PSNR"]); extra_cols.append(C_AMFM)

        x = np.arange(len(names) + len(extra_names))
        ax.bar(x, vals + extra_vals, color=colors + extra_cols)
        ax.set_xticks(x)
        ax.set_xticklabels(names + extra_names, rotation=30, ha="right", fontsize=9)
        ax.axhline(paper, color=C_PAPER, ls="--", lw=1.3)
        ax.text(0.02, paper + 0.15, f"base paper LPUDC {paper:.2f} dB", color=C_PAPER, fontsize=8.5,
                transform=ax.get_yaxis_transform())
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(f"{ds} — test-set PSNR", fontweight="bold")
        ax.set_ylim(0, max(vals + extra_vals) * 1.15)
    fig.suptitle("PSNR vs reported state of the art (grey = literature, purple = base paper's LPUDC,\n"
                 "blue/orange = ours trained on this laptop's budget)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig3_psnr_comparison.png")


# --------------------------------------------------------------------------- #
# Figure 4 — full metric comparison
# --------------------------------------------------------------------------- #
def fig4_metrics():
    sp, sc = summary("amfm_warm"), summary("dgf_cont")
    ap, ac = summary("abl_amfm"), summary("abl_dgf")
    metrics = ["PSNR", "SSIM", "LPIPS", "DISTS"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, m in zip(axes, metrics):
        labels, vals, cols, groups = [], [], [], []
        for ds, off in (("poled", 0.0), ("toled", 3.4)):
            labels.append("paper LPUDC"); vals.append((PAPER_POLED if ds == "poled" else PAPER_TOLED)[m]); cols.append(C_PAPER)
            if sc: labels.append("ours DGF"); vals.append(sc[ds][m]); cols.append(C_DGF)
            if sp: labels.append("ours AMFM"); vals.append(sp[ds][m]); cols.append(C_AMFM)
            if ac: labels.append("AMFM (4k)"); vals.append(ac[ds][m]); cols.append("#F0B27A")
            if ac: labels.append("DGF (4k)"); vals.append(ac[ds][m]); cols.append("#90B4DE")
            groups.append((ds.upper(), off, len(labels) - (5 if (ac and sc and sp) else 3)))
        x = np.arange(len(labels))
        ax.bar(x, vals, color=cols)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
        ax.set_title(m, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.18)
        # dataset separators
        for ds, off, n in groups:
            ax.axvspan(off - 0.5, off + n - 0.5, color="#F2F2F2", zorder=0)
        ax.text(1.0, max(vals) * 1.10, "POLED", ha="center", fontsize=9, color="#555555")
        if len(groups) > 1:
            ax.text(4.4, max(vals) * 1.10, "TOLED", ha="center", fontsize=9, color="#555555")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_PAPER, C_DGF, C_AMFM, "#F0B27A", "#90B4DE")],
               labels=["paper LPUDC (reference)", "ours DGF (warm, control)", "ours AMFM (warm, proposed)",
                       "AMFM (4k from scratch)", "DGF (4k from scratch)"],
               loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("Full metric comparison on the test set (higher is better for PSNR/SSIM, lower for LPIPS/DISTS)",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig4_metrics.png")


def main():
    print("Generating requested figures...")
    fig1_udc_degradation()
    fig2_amfm_architecture()
    fig3_psnr_comparison()
    fig4_metrics()


if __name__ == "__main__":
    main()
