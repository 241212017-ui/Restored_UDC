"""Warm-start an AMFM model from a trained Deep-Guided-Filtering checkpoint.

AMFM contains DGF as its core (identical parameter names: ``predict``,
``to_ab``, ``refine``) plus a zero-initialised attention correction branch.
Loading a trained DGF therefore gives an AMFM model that *is* DGF at step 0,
after which the attention can only add to it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.model import build_model
from src.utils import count_parameters_m

ROOT = Path("D:/UDC_AMFM_Project")


def main() -> None:
    src = ROOT / "outputs" / "checkpoints" / "dgf_final" / "best_psnr.pth"
    out = ROOT / "outputs" / "checkpoints" / "amfm_warm_init"
    out.mkdir(parents=True, exist_ok=True)

    state = torch.load(src, map_location="cpu")
    config = state["config"]
    config["model"]["fusion"] = "amfm"
    amfm = build_model(config["model"])
    missing, unexpected = amfm.load_state_dict(state["model"], strict=False)
    print(f"loaded DGF weights -> AMFM")
    print(f"  params        : {count_parameters_m(amfm):.4f} M")
    print(f"  missing (new) : {len(missing)} -> {sorted(set(k.split('.')[0] for k in missing))}")
    print(f"  unexpected    : {len(unexpected)}")

    # sanity: warm-started AMFM must reproduce the trained DGF module exactly
    dgf_cfg = dict(config["model"])
    dgf_cfg["fusion"] = "dgf"
    dgf_model = build_model(dgf_cfg)
    dgf_model.load_state_dict(state["model"])
    dgf_model.eval()
    amfm.eval()
    g = torch.randn(1, config["model"]["width"], 16, 16)
    h = torch.randn(1, config["model"]["width"], 16, 16)
    with torch.no_grad():
        ref = dgf_model.fusions[0](g, h)
        got = amfm.fusions[0](g, h)
    diff = (ref - got).abs().max().item()
    print(f"  init check    : max|AMFM(warm) - DGF| = {diff:.3e} (should be ~0)")
    assert diff < 1e-5, "warm start does not reproduce DGF"

    torch.save({"model": amfm.state_dict(), "config": config}, out / "init.pth")
    print(f"saved {out / 'init.pth'}")


if __name__ == "__main__":
    main()
