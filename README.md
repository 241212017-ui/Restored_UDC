# UDC-AMFM: Under-Display Camera Image Restoration

**An Attention-Based Multi-Scale Fusion Approach for Under-Display Camera Image Restoration**

UDC-AMFM is an image restoration project focused on improving the visual quality of images captured through Under-Display Cameras (UDCs). These cameras often produce degraded images due to light attenuation, diffraction, and interference from the display panel positioned above the camera sensor.

This project builds upon the **LPUDC** framework proposed by Ding et al. (*Neurocomputing*, Volume 655, 2025). It introduces a proposed **Attention-based Multi-Scale Fusion Module (AMFM)** as an alternative to the original Deep Guided Filtering (DGF) fusion stage.

The objective is to investigate whether channel and spatial attention can improve feature fusion and image restoration quality while operating within the computational limitations of a consumer laptop.

All training, ablation experiments, and evaluations were conducted locally on an **NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM**.

---

## 1. Project Overview

The restoration network uses a Laplacian pyramid to separate image information into different frequency components. These components are processed at multiple scales and subsequently combined to reconstruct the restored image.

The main contribution is AMFM, which introduces channel and spatial attention into the feature fusion process.

### Proposed AMFM Architecture

```text
       High-Frequency Features
                  |
                  |
                  +----------+
                             |
                             v
                         Concatenate
                             |
                             v
                      Channel Attention
                             |
                             v
                      Spatial Attention
                             |
                             v
                         Convolution
                             |
                             v
                      Residual Connection
                             |
                             v
                      Fused Features
```

Unlike a standalone attention-based fusion layer, AMFM is designed as an extension of the original DGF module. Its attention correction branch is zero-initialised, allowing the network to begin with the exact DGF behaviour before learning additional corrections.

Architecture visualisations are available in:

- `report/figures/fig_architecture.png`
- `report/figures/fig_amfm.png`

---

## 2. Key Features

- **Laplacian Pyramid Restoration:** Processes image information across multiple frequency scales.
- **Attention-Based Fusion:** Combines channel and spatial attention to refine multi-scale feature representations.
- **Baseline Comparison:** Evaluates AMFM against DGF, concatenation-based fusion, and a model without an explicit fusion module.
- **Matched-Budget Ablation:** Compares different fusion strategies under controlled training conditions.
- **Memory-Efficient Training:** Uses mixed-precision training, 256 × 256 patches, and cached crops to accommodate 4 GB GPU memory.
- **Full-Resolution Evaluation:** Measures PSNR and SSIM on original 1024 × 2048 images.
- **Perceptual Evaluation:** Includes LPIPS and DISTS to assess perceptual image quality.
- **Interactive Demo:** Provides a Streamlit interface for testing image restoration.
- **Reproducible Experiments:** Scripts generate model checkpoints, training logs, evaluation results, and visualisations locally.

---

## 3. Repository Structure

```text
UDC-AMFM/
│
├── src/
│   ├── models and network architecture
│   ├── Laplacian pyramid implementation
│   ├── AMFM, DGF and alternative fusion blocks
│   ├── loss functions and evaluation metrics
│   ├── training and evaluation pipelines
│   └── dataset utilities
│
├── configs/
│   └── YAML configurations for experiments
│
├── scripts/
│   ├── prepare_data.py
│   ├── build_crop_cache.py
│   ├── benchmark scripts
│   ├── run_all.ps1
│   └── figure generation
│
├── app/
│   └── streamlit_app.py
│
├── report/
│   ├── UDС_AMFM_Report.md
│   ├── UDС_AMFM_Report.pdf
│   └── make_report.py
│
├── slides/
│   ├── UDС_AMFM_Presentation.pptx
│   └── make_slides.py
│
├── tests/
│   └── Unit tests for model, pyramid, losses and metrics
│
├── requirements.txt
└── README.md
```

The datasets and generated experiment outputs are not included in this repository. Place the datasets under `data/` and generated files will be written under `outputs/`.

---

## 4. Dataset

The experiments use POLED and TOLED under-display camera datasets.

Both datasets are divided into training, validation, and test subsets.

| Split | POLED | TOLED | Total |
|---|---:|---:|---:|
| Training | 240 | 240 | 480 |
| Validation | 30 | 30 | 60 |
| Testing | 30 | 30 | 60 |

**Image resolution:** 1024 × 2048 pixels.

The datasets are not included in this repository. Obtain them separately and place the paired training images under `data/Train/{Poled,Toled}/{LQ,HQ}/`, with the validation and test `.mat` files directly under `data/`. The provided preprocessing scripts convert the `.mat` files into memory-mapped `.npy` arrays and cache training crops to reduce repeated data-loading overhead.

---

## 5. Models and Experimental Setup

Four fusion configurations are evaluated through a matched-budget ablation study.

| Model | Fusion Strategy |
|---|---|
| `abl_none` | No explicit fusion module |
| `abl_concat` | Concatenation-based fusion |
| `abl_dgf` | Deep Guided Filtering |
| `abl_amfm` | Proposed Attention-based Multi-Scale Fusion |

Two additional experiments investigate the behaviour of the DGF and AMFM models under a longer warm-start training schedule.

- `dgf_final`: DGF baseline.
- `amfm_final`: Proposed AMFM model.

The experiments use the same dataset splits and corresponding training budgets to make the comparisons meaningful.

### Evaluation Metrics

| Metric | Purpose |
|---|---|
| PSNR | Measures pixel-level reconstruction quality |
| SSIM | Evaluates structural similarity |
| LPIPS | Measures perceptual similarity using deep features |
| DISTS | Evaluates structural and texture-based perceptual similarity |

PSNR and SSIM are calculated at the original image resolution. LPIPS and DISTS use images resized along their longer dimension to 1024 pixels to accommodate GPU memory constraints.

---

## 6. Experimental Results

The final evaluation covers **60 test images at their original resolution of 1024 × 2048 pixels**.

### Matched-Budget Ablation

| Model | POLED PSNR | TOLED PSNR | Combined PSNR | SSIM | LPIPS | DISTS |
|---|---:|---:|---:|---:|---:|---:|
| `abl_dgf` | 22.91 | 23.73 | 23.32 | 0.7705 | 0.3488 | 0.2354 |
| `abl_amfm` | **23.06** | 23.54 | 23.30 | **0.7778** | **0.3390** | **0.2285** |

*The ablation results correspond to the reported 4k-iteration evaluation checkpoints.*

### Warm-Start Comparison

| Model | POLED PSNR | TOLED PSNR | Combined PSNR | SSIM | LPIPS | DISTS |
|---|---:|---:|---:|---:|---:|---:|
| `dgf_cont` | **23.00** | 25.88 | **24.44** | **0.7609** | **0.3400** | **0.2227** |
| `amfm_warm` | 22.81 | **25.90** | 24.35 | 0.7600 | 0.3404 | 0.2230 |

*The warm-start comparison uses an additional 5k training iterations.*

### What the Results Show

The experiments indicate that AMFM is competitive with the original DGF fusion strategy under the tested computational budgets.

In the matched-budget ablation, AMFM performs better in 9 out of 12 reported metric comparisons, including all three POLED metrics and the combined SSIM, LPIPS, and DISTS measurements.

In the longer warm-start comparison, DGF performs better in 8 of the 12 comparisons, while AMFM records a small improvement in TOLED PSNR.

The combined PSNR difference remains small:

- **4k ablation:** 0.02 dB difference.
- **Warm-start comparison:** 0.09 dB difference.

Overall, the results suggest that attention-based fusion can provide competitive restoration quality and improvements in selected perceptual metrics. However, the experiments do not establish a consistent PSNR improvement over DGF.

---

## 7. AMFM: Proposed Contribution

The primary contribution of this project is the Attention-based Multi-Scale Fusion Module.

The original LPUDC framework uses Deep Guided Filtering for feature fusion. This project extends that stage with an attention-based correction branch.

The proposed module combines:

1. Feature concatenation to bring multi-scale information together.
2. Channel attention to adaptively refine feature-channel responses.
3. Spatial attention to emphasise relevant spatial information.
4. Convolution to transform the refined feature representation.
5. A residual connection to preserve the original fusion behaviour.

### DGF Compatibility

An important design consideration is the initialisation of the attention correction branch.

The branch is zero-initialised, ensuring that the initial AMFM output is identical to the DGF output.

The implementation was verified numerically:

```text
max|AMFM(init) - DGF| = 0.0
```

This provides an exact starting point for warm-start experiments and allows the attention branch to learn corrections without initially changing the baseline behaviour.

---

## 8. Installation and Usage

### Requirements

- Python
- PyTorch
- CUDA-compatible NVIDIA GPU
- Dependencies listed in `requirements.txt`

Install the dependencies:

```powershell
pip install -r requirements.txt
```

### Step 1: Prepare the Dataset

After placing the datasets in the expected `data/` layout described above, prepare the validation/test arrays and training crop cache:

```powershell
python scripts/prepare_data.py
python scripts/build_crop_cache.py
```

These scripts convert the source `.mat` files into memory-mapped `.npy` arrays and generate precomputed 256 × 256 training crops.

### Step 2: Run Training

Execute the complete experiment schedule:

```powershell
powershell -File scripts/run_all.ps1
```

This runs the matched fusion ablations and the longer DGF and AMFM experiments.

### Step 3: Evaluate a Model

For example, to evaluate the proposed AMFM model:

```powershell
python -m src.evaluate --config configs/amfm.yaml `
    --checkpoint outputs/checkpoints/amfm_final/best_psnr.pth `
    --name amfm_final --tta
```

The evaluation pipeline generates per-image metrics and aggregate test results.

### Step 4: Generate Figures and Reports

```powershell
python scripts/make_figures.py
python report/make_report.py
python slides/make_slides.py
```

### Step 5: Launch the Interactive Demo

```powershell
streamlit run app/streamlit_app.py
```

### Step 6: Run Unit Tests

```powershell
pytest -q tests
```

---

## 9. Hardware and Memory Optimisation

A major practical constraint during development was the GPU's 4 GB VRAM capacity.

Rather than relying on high-end hardware, the training and evaluation pipelines were configured to operate within the available resources.

| Component | Configuration |
|---|---|
| GPU | NVIDIA RTX 3050 Laptop GPU |
| VRAM | 4 GB |
| Training patches | 256 × 256 |
| Training precision | Automatic Mixed Precision (AMP) |
| Inference | 512 × 512 tiled processing |
| Dataset storage | Memory-mapped NumPy arrays |
| Crop preparation | Precomputed crop cache |

These optimisations make it possible to conduct multi-scale restoration experiments on a consumer laptop.

---

## 10. Reproducibility and Research Limitations

The original LPUDC paper reports substantially higher PSNR values:

- POLED: 33.99 dB
- TOLED: 38.62 dB

These results were obtained using approximately 300,000 full-resolution iterations on an NVIDIA RTX 3090 with 24 GB VRAM.

This project operates under a considerably smaller computational budget, using a 4 GB laptop GPU and a different training schedule. Therefore, its absolute metrics should not be directly compared with the paper's reported values as if they were obtained under identical conditions.

The main objective is to study the relative behaviour of different fusion strategies under matched experimental settings.

Additional considerations:

- All reported evaluation metrics are calculated from model predictions.
- Results are generated from evaluation outputs rather than hard-coded values.
- Checkpoints, logs, and per-image evaluation CSV files are generated locally under `outputs/` and are not included in the repository.
- LPIPS and DISTS are evaluated at a memory-adjusted resolution.
- The experiments are limited to the available dataset size and training budget.

The project report includes detailed evaluation tables, ablation results, per-image analysis, attention visualisations, and further discussion.

---

## 11. Documentation and Visualisations

Additional project materials are available in the repository:

| Resource | Location |
|---|---|
| Full research report | `report/UDC_AMFM_Report.pdf` |
| Report source | `report/UDC_AMFM_Report.md` |
| Presentation | `slides/UDC_AMFM_Presentation.pptx` |
| Network architecture | `report/figures/fig_architecture.png` |
| AMFM architecture | `report/figures/fig_amfm.png` |
| Model checkpoints | Generated locally under `outputs/checkpoints/` |
| Evaluation results | Generated locally under `outputs/eval/` |

---

## 12. Future Improvements

Potential directions for further investigation include:

- Increasing the training budget and evaluating convergence behaviour.
- Testing the proposed fusion module on larger and more diverse datasets.
- Investigating alternative channel and spatial attention mechanisms.
- Exploring lightweight architectures for faster inference.
- Evaluating generalisation across different under-display camera configurations.
- Conducting additional statistical analysis across multiple training seeds.

---

## Acknowledgements

This project is based on the LPUDC research framework proposed by Ding et al., published in *Neurocomputing*, Volume 655, 2025.

The work explores an alternative attention-based feature fusion strategy while retaining the original framework as the experimental baseline.

**Project:** UDC-AMFM — Under-Display Camera Image Restoration
**Core Contribution:** Attention-based Multi-Scale Fusion Module (AMFM)
**Research Focus:** Image Restoration · Laplacian Pyramids · Multi-Scale Learning · Attention Mechanisms · Deep Learning
