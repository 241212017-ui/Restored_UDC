"""Streamlit demo: restore an under-display-camera image with LPUDC-AMFM.

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import tiled_predict  # noqa: E402
from src.metrics import psnr, ssim  # noqa: E402
from src.model import build_model  # noqa: E402

st.set_page_config(page_title="UDC Restoration — LPUDC + AMFM", layout="wide")


@st.cache_resource(show_spinner=False)
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for name in ("amfm_warm", "dgf_cont"):
        path = ROOT / "outputs" / "checkpoints" / name / "best_psnr.pth"
        if path.exists():
            state = torch.load(path, map_location="cpu")
            m = build_model(state["config"]["model"]).to(device).eval()
            m.load_state_dict(state["model"])
            models[name] = m
    return models, device


def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


st.title("Under-Display Camera Image Restoration")
st.caption("LPUDC with the proposed Attention-based Multi-Scale Fusion Module (AMFM)")

models, device = load_models()
if not models:
    st.error("No checkpoints found. Train first (scripts/run_all.ps1).")
    st.stop()

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", sorted(models.keys(), key=lambda n: ("amfm" not in n, n)))
    st.write(f"Device: `{device}`")
    st.write(f"Params: `{sum(p.numel() for p in models[model_name].parameters()) / 1e6:.3f} M`")
    st.divider()
    st.markdown("**Try:** upload a degraded UDC image, or use a built-in POLED/TOLED sample.")
    sample_ds = st.selectbox("Built-in sample", ["poled", "toled"])
    sample_idx = st.slider("sample index", 0, 29, 3)

uploaded = st.file_uploader("Upload a degraded (LQ) image", type=["png", "jpg", "jpeg", "bmp"])
gt = None
if uploaded is not None:
    lq_img = Image.open(uploaded)
    st.info("Uploaded image has no ground truth, so PSNR/SSIM are not shown.")
else:
    disp = np.load(ROOT / "data" / f"{sample_ds}_test_display.npy", mmap_mode="r")[sample_idx]
    gtarr = np.load(ROOT / "data" / f"{sample_ds}_test_gt.npy", mmap_mode="r")[sample_idx]
    lq_img = Image.fromarray(np.asarray(disp))
    gt = to_tensor(Image.fromarray(np.asarray(gtarr)))

col1, col2 = st.columns(2)
with col1:
    st.subheader("Degraded input")
    st.image(lq_img, use_container_width=True)
with col2:
    st.subheader("Restored")
    with st.spinner("Restoring..."):
        lq = to_tensor(lq_img).to(device)
        pred = tiled_predict(models[model_name], lq)
        st.image(to_pil(pred), use_container_width=True)

if gt is not None:
    gt = gt.to(device)
    c1, c2, c3 = st.columns(3)
    c1.metric("PSNR", f"{psnr(pred, gt).item():.2f} dB")
    c2.metric("SSIM", f"{ssim(pred, gt).item():.4f}")
    c3.metric("Input PSNR", f"{psnr(lq, gt).item():.2f} dB")
    with st.expander("Ground truth"):
        st.image(to_pil(gt), use_container_width=True)

st.divider()
st.markdown(
    "**How it works.** The image is decomposed into a 3-level Laplacian pyramid. A low-frequency "
    "encoder-decoder fixes global brightness/colour, dilated residual stacks enhance each "
    "high-frequency band, and the **AMFM** module fuses adjacent scales using channel + spatial "
    "attention before the pyramid is summed back into the restored image."
)
