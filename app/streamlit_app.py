
"""
Streamlit Demo: Under-Display Camera Image Restoration
LPUDC + Attention-based Multi-Scale Fusion Module (AMFM)

Run locally:
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

# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints"
DATA_DIR = ROOT / "data"

# ---------------------------------------------------------
# PROJECT IMPORTS
# ---------------------------------------------------------

from src.evaluate import tiled_predict
from src.metrics import psnr, ssim
from src.model import build_model


# ---------------------------------------------------------
# STREAMLIT CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="UDC Image Restoration | LPUDC + AMFM",
    page_icon="📷",
    layout="wide",
)

st.title("Under-Display Camera Image Restoration")

st.caption(
    "LPUDC with the proposed Attention-based "
    "Multi-Scale Fusion Module (AMFM)"
)


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_models():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    models = {}
    diagnostics = []

    checkpoint_names = (
        "amfm_warm",
        "dgf_cont",
    )

    for name in checkpoint_names:

        checkpoint_path = (
            CHECKPOINT_DIR
            / name
            / "best_psnr.pth"
        )

        # Check whether checkpoint exists
        if not checkpoint_path.is_file():

            diagnostics.append(
                f"MISSING: {checkpoint_path}"
            )

            continue

        try:

            # Load checkpoint
            state = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

            # Validate checkpoint structure
            if "config" not in state or "model" not in state:
                raise ValueError(
                    "Checkpoint does not contain "
                    "'config' and 'model' keys."
                )

            # Build model from saved configuration
            model = build_model(
                state["config"]["model"]
            )

            model.load_state_dict(
                state["model"]
            )

            model = model.to(device).eval()

            models[name] = model

            size_mb = (
                checkpoint_path.stat().st_size
                / (1024 * 1024)
            )

            diagnostics.append(
                f"LOADED: {name} ({size_mb:.2f} MB)"
            )

        except Exception as error:

            diagnostics.append(
                f"FAILED: {name}: "
                f"{type(error).__name__}: {error}"
            )

    return models, device, diagnostics


# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------

with st.spinner("Loading trained restoration models..."):

    models, device, diagnostics = load_models()


# ---------------------------------------------------------
# CHECKPOINT DIAGNOSTICS
# ---------------------------------------------------------

if not models:

    st.error(
        "No restoration models could be loaded."
    )

    st.subheader("Deployment Diagnostics")

    st.write("Project root:", str(ROOT))

    st.write(
        "Checkpoint directory:",
        str(CHECKPOINT_DIR),
    )

    st.write(
        "Checkpoint directory exists:",
        CHECKPOINT_DIR.exists(),
    )

    st.write("Available checkpoint files:")

    if CHECKPOINT_DIR.exists():

        checkpoint_files = list(
            CHECKPOINT_DIR.rglob("*.pth")
        )

        if checkpoint_files:

            for file in checkpoint_files:
                st.code(
                    str(file.relative_to(ROOT))
                )

        else:
            st.warning(
                "No .pth files were found in the checkpoint directory."
            )

    else:

        st.warning(
            "The outputs/checkpoints directory does not exist."
        )

    st.subheader("Loading Results")

    for message in diagnostics:
        st.code(message)

    st.info(
        "Check the GitHub commit deployed by Streamlit Cloud "
        "and verify that the checkpoint files are included."
    )

    st.stop()


# ---------------------------------------------------------
# MODEL SELECTION
# ---------------------------------------------------------

with st.sidebar:

    st.header("Restoration Settings")

    model_name = st.selectbox(
        "Select restoration model",
        sorted(
            models.keys(),
            key=lambda name: (
                "amfm" not in name,
                name,
            ),
        ),
    )

    st.write(f"Device: `{device}`")

    parameter_count = sum(
        parameter.numel()
        for parameter in models[model_name].parameters()
    )

    st.write(
        f"Parameters: `{parameter_count / 1e6:.3f} M`"
    )

    st.divider()

    st.subheader("Available Models")

    for message in diagnostics:
        st.write(message)


# ---------------------------------------------------------
# IMAGE CONVERSION
# ---------------------------------------------------------

def to_tensor(image: Image.Image) -> torch.Tensor:

    array = np.asarray(
        image.convert("RGB"),
        dtype=np.float32,
    ) / 255.0

    tensor = torch.from_numpy(
        array.transpose(2, 0, 1)
    ).unsqueeze(0)

    return tensor


def to_pil(tensor: torch.Tensor) -> Image.Image:

    array = (
        tensor.clamp(0, 1)[0]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        * 255
    ).astype(np.uint8)

    return Image.fromarray(array)


# ---------------------------------------------------------
# INPUT IMAGE
# ---------------------------------------------------------

st.header("Restore Your Image")

uploaded_file = st.file_uploader(
    "Upload a degraded under-display-camera image",
    type=["png", "jpg", "jpeg", "bmp"],
)

ground_truth = None
lq_image = None

if uploaded_file is not None:

    try:

        lq_image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.success("Image uploaded successfully.")

        st.info(
            "Uploaded images do not include ground truth. "
            "PSNR and SSIM will not be calculated."
        )

    except Exception as error:

        st.error(
            f"Unable to open image: {error}"
        )

        st.stop()


# ---------------------------------------------------------
# BUILT-IN DATASET SAMPLES
# ---------------------------------------------------------

else:

    st.subheader("Built-in Test Samples")

    sample_options = {
        "POLED": "poled",
        "TOLED": "toled",
    }

    sample_type = st.selectbox(
        "Select dataset",
        list(sample_options.keys()),
    )

    sample_name = sample_options[sample_type]

    display_path = (
        DATA_DIR
        / f"{sample_name}_test_display.npy"
    )

    ground_truth_path = (
        DATA_DIR
        / f"{sample_name}_test_gt.npy"
    )

    if (
        not display_path.is_file()
        or not ground_truth_path.is_file()
    ):

        st.warning(
            "Built-in dataset samples are unavailable "
            "in this deployment. Upload a degraded image "
            "to use the restoration model."
        )

        st.stop()

    try:

        display_data = np.load(
            display_path,
            mmap_mode="r",
        )

        ground_truth_data = np.load(
            ground_truth_path,
            mmap_mode="r",
        )

        if len(display_data) == 0:

            st.error("The selected test dataset is empty.")
            st.stop()

        sample_idx = st.slider(
            "Sample index",
            min_value=0,
            max_value=min(
                len(display_data),
                len(ground_truth_data),
            ) - 1,
            value=min(
                3,
                min(
                    len(display_data),
                    len(ground_truth_data),
                ) - 1,
            ),
        )

        lq_image = Image.fromarray(
            np.asarray(display_data[sample_idx])
        ).convert("RGB")

        ground_truth = to_tensor(
            Image.fromarray(
                np.asarray(ground_truth_data[sample_idx])
            )
        )

    except Exception as error:

        st.error(
            f"Unable to load dataset sample: {error}"
        )

        st.stop()


# ---------------------------------------------------------
# RESTORATION
# ---------------------------------------------------------

if lq_image is not None:

    st.divider()

    st.header("Restoration Results")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Degraded Input")

        st.image(
            lq_image,
            width="stretch",
        )

    with col2:

        st.subheader("Restored Output")

        try:

            selected_model = models[model_name]

            with st.spinner(
                f"Restoring image using {model_name}..."
            ):

                input_tensor = to_tensor(
                    lq_image
                ).to(device)

                with torch.inference_mode():

                    prediction = tiled_predict(
                        selected_model,
                        input_tensor,
                    )

                restored_image = to_pil(
                    prediction
                )

            st.image(
                restored_image,
                width="stretch",
            )

            st.success(
                f"Restoration completed using {model_name}."
            )

            # Download restored image
            import io

            image_buffer = io.BytesIO()

            restored_image.save(
                image_buffer,
                format="PNG",
            )

            st.download_button(
                label="Download Restored Image",
                data=image_buffer.getvalue(),
                file_name="restored_udc_image.png",
                mime="image/png",
                width="stretch",
            )

        except Exception as error:

            st.error(
                f"Image restoration failed: "
                f"{type(error).__name__}: {error}"
            )

            st.stop()


# ---------------------------------------------------------
# IMAGE QUALITY METRICS
# ---------------------------------------------------------

if ground_truth is not None:

    st.divider()

    st.header("Image Quality Evaluation")

    try:

        gt = ground_truth.to(device)

        with torch.inference_mode():

            restored_psnr = psnr(
                prediction,
                gt,
            ).item()

            restored_ssim = ssim(
                prediction,
                gt,
            ).item()

            input_psnr = psnr(
                input_tensor,
                gt,
            ).item()

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Restored PSNR",
            f"{restored_psnr:.2f} dB",
        )

        metric2.metric(
            "Restored SSIM",
            f"{restored_ssim:.4f}",
        )

        metric3.metric(
            "Input PSNR",
            f"{input_psnr:.2f} dB",
        )

        with st.expander("View Ground Truth"):

            st.image(
                to_pil(gt),
                width="stretch",
            )

    except Exception as error:

        st.warning(
            f"Unable to calculate image metrics: {error}"
        )


# ---------------------------------------------------------
# MODEL DESCRIPTION
# ---------------------------------------------------------

st.divider()

st.header("About the Restoration Architecture")

st.markdown(
    """
    The restoration pipeline is based on LPUDC and the proposed
    **Attention-based Multi-Scale Fusion Module (AMFM)**.

    The architecture includes:

    1. A three-level Laplacian pyramid for image decomposition.
    2. A low-frequency encoder-decoder for global brightness
       and colour correction.
    3. Dilated residual stacks for high-frequency enhancement.
    4. Channel and spatial attention for multi-scale feature fusion.
    5. Pyramid reconstruction to generate the restored image.

    Select AMFM to evaluate the proposed attention-based model,
    or DGF to compare against the baseline.
    """
)

st.caption(
    "UDC-AMFM | Research Demonstration"
)
