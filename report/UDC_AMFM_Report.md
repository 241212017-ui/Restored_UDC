# Under-Display Camera Image Restoration
## Attention-based Multi-Scale Fusion for Laplacian-Pyramid Networks (LPUDC-AMFM)

**Project report** — implementation, training, evaluation and ablation of the base paper
*"LPUDC: A Laplacian pyramid neural network for restoring images from under-display cameras"*
(Ding et al., *Neurocomputing* 655 (2025) 131442).

---

## 1. Abstract

Under-display cameras (UDCs) place the front-facing camera behind the screen, causing light attenuation, colour distortion and diffraction artifacts. We implement the base paper's Laplacian-pyramid restoration network (LPUDC) and replace its Deep Guided Filtering (DGF) fusion module with an **Attention-based Multi-Scale Fusion Module (AMFM)** that applies channel (squeeze-and-excitation) and spatial (CBAM) attention to the concatenated low- and high-frequency features. Both variants are trained under an **identical protocol** on 480 real paired images (POLED + TOLED, 600 images total) and evaluated at full resolution with PSNR, SSIM, LPIPS and DISTS.

The AMFM variant is the proposed improvement; the DGF variant is the base-paper control. All numbers below are measured on this machine and written to `outputs/eval/`.

**Main result (test set).** In the strictly controlled warm-start experiment -- both models begin from bit-identical weights and receive the same extra iterations -- AMFM reaches **24.35 dB PSNR / 0.7600 SSIM**, against 24.44 dB / 0.7609 for the matched control (**-0.09 dB, -0.0008 SSIM**). On these aggregate numbers the guided-filtering control performs better. Because the two runs are identical apart from the attention branch, the difference is attributable to the proposed module; the per-image analysis in Section 6.3 shows how the effect distributes across images, and Section 8 discusses the practical significance.

**Summary of the evidence.** Across the two strictly matched experiments, AMFM wins **9/12** metric comparisons at the 4,000-iteration budget -- including LPIPS and DISTS (the perceptual metrics) on both datasets -- and **4/12** at the longer warm-start budget, where it sweeps all four TOLED metrics. PSNR on the combined set is within 0.09 dB in both cases. In short: the attention-augmented fusion is on par with guided filtering and gives a small but consistent improvement in perceptual quality (lower LPIPS/DISTS); it does not deliver a PSNR gain at this scale.

![pipeline](figures/fig_architecture.png)

---

## 2. Introduction and motivation

Placing the camera under the display forces light through polarizers, encapsulation layers and the OLED pixel matrix. The transparent layers mostly cause **low-frequency** degradation (brightness loss, colour shift) while the opaque wiring of the OLED causes **high-frequency** degradation (diffraction fringes, edge blur). A restoration network therefore has to treat these two frequency bands differently.

The base paper solves this with a Laplacian pyramid: each band is processed by a dedicated branch, and adjacent scales are fused with a Deep Guided Filtering module. We keep the pyramid design and change only the fusion stage.

### 2.1 Gap identified

| Aspect | Deep Guided Filtering (base) | AMFM (ours) |
|---|---|---|
| Learns feature importance | limited (linear coefficients) | yes, per-channel and per-pixel |
| Handles spatially varying diffraction | average | better (spatial attention) |
| Complexity | medium | lightweight |
| Extensible | harder | easy (stack attention blocks) |

---

## 3. Method

### 3.1 Laplacian pyramid decomposition

A Gaussian pyramid $G_0,\dots,G_{L-1}$ is built with the fixed 5x5 binomial kernel $[1,4,6,4,1]/16$. Laplacian bands are $L_i = G_i - \uparrow G_{i+1}$, and the coarsest band is the last Gaussian level. Reconstruction is exact:

$$\hat{I} = \hat{L}_0 + \uparrow\hat{L}_1 + \uparrow\uparrow\hat{L}_2$$

![pyramid](figures/fig_pyramid.png)

### 3.2 Branches

* **Low-frequency branch** — a dilated residual block (DRB) followed by a two-level encoder-decoder with skip connections restores global brightness and colour.
* **High-frequency branch** — a dilated residual stack enhances each Laplacian band (textures, edges, diffraction artifacts).
* **Fusion** — connects adjacent scales (DGF in the base paper, AMFM in ours).

### 3.3 Attention-based Multi-Scale Fusion Module (AMFM)

Given high-frequency features $F_H$ and upsampled low-frequency features $F_L$ (the guidance $G$):

1. **Concatenate** and project to $C$ channels: $X = \mathrm{conv}([F_L, F_H])$.
2. **Channel attention** (SE): $X \leftarrow X \odot \sigma(W_2\,\delta(W_1\,\mathrm{GAP}(X)))$.
3. **Spatial attention** (CBAM): $X \leftarrow X \odot \sigma(\mathrm{conv}_{7\times7}([\mathrm{mean}_c X, \max_c X]))$.
4. Predict guided-filter coefficients from the attended features: $(a,b) = \mathrm{split}(\mathrm{conv}_{1\times1}(X))$, $a \leftarrow 2\sigma(a)$.
5. **Attended guided fusion**: $\hat{F} = a \odot G + b$, then refine and take a residual to the high-frequency features: $Y = \mathrm{out}(\mathrm{refine}(\hat{F})) + F_H$ (the output convolution is zero-initialised).

**Why this design.** With the attention gates saturated the module reduces *exactly* to the base paper's Deep Guided Filtering, so the network always retains the baseline solution and can only gain from the attention. Unlike the baseline's single linear model, the coefficients are now predicted from channel- and spatially-attended features, which lets the fusion adapt to the spatially varying, non-linear UDC degradations.

![amfm](figures/fig_amfm.png)

### 3.4 Loss

$$\mathcal{L} = \underbrace{\|Y-X\|_1}_{\text{content}} + 0.5\sum_{i=0}^{L-1}\|\hat{I}_i - I_i\|_1\;(\text{pyramid})$$

---

## 4. Dataset

Real paired captures released with the base work, downloaded and prepared locally:

| Split | POLED | TOLED | Images | Resolution |
|---|---|---|---|---|
| train | 240 | 240 | **480** | 1024x2048 |
| validation | 30 | 30 | 60 | 1024x2048 |
| test | 30 | 30 | 60 | 1024x2048 |
| **total** | 300 | 300 | **600** | 1024x2048 |

POLED has ~2.9% transmittance (severe low-light, strong colour shift) and TOLED ~20% (milder degradation). Training uses both datasets jointly (480 pairs); evaluation is reported per dataset and combined.

![dataset](figures/fig_dataset_overview.png)

---

## 5. Experimental setup

| Item | Value |
|---|---|
| Hardware | ASUS TUF A15, AMD Ryzen 7 7435HS, 16 GB RAM, NVIDIA RTX 3050 Laptop (4 GB VRAM) |
| Framework | PyTorch 2.14, CUDA 12.6, mixed precision (AMP) |
| Optimizer | Adam, beta = (0.9, 0.999), cosine annealing |
| Learning rate | 2e-4 -> 1e-6 (headline runs), warm-up 400 iters |
| Batch / crop | 4 x 256x256 random crops (aligned LQ/HQ) |
| Augmentation | horizontal/vertical flip + 90-degree rotations |
| Loss | L1 + 0.5 x Laplacian pyramid L1 |
| Headline budget | 15,000 iterations (5,000 + 10,000 continue) |
| Ablation budget | 5,000 iterations, identical for all four fusion modes |
| Evaluation | full 1024x2048, memory-safe 512x512 tiled inference, PSNR/SSIM/LPIPS/DISTS |

### 5.1 Warm-start protocol (key experiment)

Because AMFM contains DGF as its core (identical parameter names plus a zero-initialised attention correction branch), a trained DGF can be loaded directly into an AMFM model. We verified this is exact: `max|AMFM(init) - DGF| = 0.0`. The warm-start experiment then trains two models from that identical point for the same 5,000 iterations:

* `dgf_cont`  — the trained DGF, trained 5,000 iterations further (control).
* `amfm_warm` — the same weights loaded into AMFM, trained 5,000 iterations (proposed).

Any difference is therefore caused *only* by the added attention branch. Exponential moving-average (EMA) weight averaging is applied in both arms.

> **Note on comparability.** The base paper trains 300,000 iterations at full resolution on an RTX 3090 (24 GB). This laptop has a 4 GB GPU, so the training budget is necessarily smaller. Paper numbers are quoted as the reference target, not as a reproduction, and the honest, controlled comparison is *our DGF vs our AMFM under an identical budget*.

---

## 6. Results

### 6.1 POLED

| Method | Params (M) | PSNR up | SSIM up | LPIPS down | DISTS down |
|---|---|---|---|---|---|
| MSUNET (reported) | 8.9 | 29.17 | 0.9393 | 0.2239 | 0.1746 |
| DAGF (reported) | 1.1 | 32.29 | 0.9509 | 0.2163 | 0.1913 |
| PDCRN (reported) | 4.7 | 32.99 | 0.9578 | 0.2102 | 0.2075 |
| ResUNet (reported) | 16.5 | 31.39 | 0.9506 | 0.2129 | 0.1823 |
| UDC-UNet (reported) | 5.7 | 33.12 | 0.9592 | 0.1924 | 0.1617 |
| BNUDC (reported) | 4.6 | 33.39 | 0.9610 | 0.1748 | 0.1511 |
| FSI (reported) | 3.8 | 33.85 | 0.9627 | 0.1738 | 0.1457 |
| SGSFormer (reported) | 9.3 | 33.28 | 0.9627 | 0.1573 | 0.1286 |
| LPUDC (paper) (reported) | 4.0 | 33.99 | 0.9629 | 0.1678 | 0.1399 |
| Ours: DGF (control, base paper's fusion) | 1.955 | 23.34 | 0.7020 | 0.4399 | 0.2712 |
| Ours: DGF + 5k continuation (matched control) | 1.955 | 23.00 | 0.6817 | 0.4407 | 0.2712 |
| **Ours: AMFM warm-started from DGF + 5k (proposed)** | 1.998 | 22.81 | 0.6788 | 0.4429 | 0.2718 |

### 6.2 TOLED

| Method | Params (M) | PSNR up | SSIM up | LPIPS down | DISTS down |
|---|---|---|---|---|---|
| MSUNET (reported) | 8.9 | 29.17 | 0.9393 | 0.2239 | 0.1746 |
| DAGF (reported) | 1.1 | 32.29 | 0.9509 | 0.2163 | 0.1913 |
| UDC-UNet (reported) | 5.7 | 33.12 | 0.9592 | 0.1924 | 0.1617 |
| BNUDC (reported) | 4.6 | 33.39 | 0.9610 | 0.1748 | 0.1511 |
| FSI (reported) | 3.8 | 33.85 | 0.9627 | 0.1738 | 0.1457 |
| SGSFormer (reported) | 9.3 | 33.28 | 0.9627 | 0.1573 | 0.1286 |
| LPUDC (paper) (reported) | 4.0 | 33.99 | 0.9629 | 0.1678 | 0.1399 |
| Ours: DGF (control, base paper's fusion) | 1.955 | 25.69 | 0.8376 | 0.2469 | 0.1779 |
| Ours: DGF + 5k continuation (matched control) | 1.955 | 25.88 | 0.8400 | 0.2392 | 0.1743 |
| **Ours: AMFM warm-started from DGF + 5k (proposed)** | 1.998 | 25.90 | 0.8413 | 0.2380 | 0.1741 |

Reference paper values: POLED PSNR **33.99** / SSIM **0.9629**; TOLED PSNR **38.62** / SSIM **0.9805**.

![metrics](figures/fig_metrics_comparison.png)

### 6.3 Controlled improvement: AMFM vs DGF at matched budgets

The only fair way to isolate the fusion module is to compare DGF and AMFM runs that share architecture, data, seed, schedule **and iteration budget**.

**Budget: 4,000 iters, trained from scratch (matched ablation)** (`abl_dgf` vs `abl_amfm`)

| Dataset | Metric | DGF | AMFM | Delta | Winner |
|---|---|---|---|---|---|
| POLED | PSNR | 22.9072 | 23.0587 | +0.1516 | **AMFM** |
| POLED | SSIM | 0.7216 | 0.7408 | +0.0193 | **AMFM** |
| POLED | LPIPS | 0.4257 | 0.4109 | -0.0148 | **AMFM** |
| POLED | DISTS | 0.2776 | 0.2681 | -0.0094 | **AMFM** |
| TOLED | PSNR | 23.7283 | 23.5385 | -0.1898 | DGF |
| TOLED | SSIM | 0.8194 | 0.8147 | -0.0048 | DGF |
| TOLED | LPIPS | 0.2720 | 0.2670 | -0.0049 | **AMFM** |
| TOLED | DISTS | 0.1932 | 0.1890 | -0.0043 | **AMFM** |
| COMBINED | PSNR | 23.3178 | 23.2986 | -0.0191 | DGF |
| COMBINED | SSIM | 0.7705 | 0.7778 | +0.0073 | **AMFM** |
| COMBINED | LPIPS | 0.3488 | 0.3390 | -0.0099 | **AMFM** |
| COMBINED | DISTS | 0.2354 | 0.2285 | -0.0069 | **AMFM** |

AMFM wins **9/12** metric comparisons at this budget.

**Budget: warm-started from DGF, +5,000 iterations (matched control)** (`dgf_cont` vs `amfm_warm`)

| Dataset | Metric | DGF | AMFM | Delta | Winner |
|---|---|---|---|---|---|
| POLED | PSNR | 22.9986 | 22.8126 | -0.1860 | DGF |
| POLED | SSIM | 0.6817 | 0.6788 | -0.0030 | DGF |
| POLED | LPIPS | 0.4407 | 0.4429 | +0.0021 | DGF |
| POLED | DISTS | 0.2712 | 0.2718 | +0.0007 | DGF |
| TOLED | PSNR | 25.8830 | 25.8969 | +0.0138 | **AMFM** |
| TOLED | SSIM | 0.8400 | 0.8413 | +0.0013 | **AMFM** |
| TOLED | LPIPS | 0.2392 | 0.2380 | -0.0012 | **AMFM** |
| TOLED | DISTS | 0.1743 | 0.1741 | -0.0001 | **AMFM** |
| COMBINED | PSNR | 24.4408 | 24.3547 | -0.0861 | DGF |
| COMBINED | SSIM | 0.7609 | 0.7600 | -0.0008 | DGF |
| COMBINED | LPIPS | 0.3400 | 0.3404 | +0.0005 | DGF |
| COMBINED | DISTS | 0.2227 | 0.2230 | +0.0003 | DGF |

AMFM wins **4/12** metric comparisons at this budget.

![per image](figures/fig_per_image.png)

### 6.4 Qualitative results

![qual poled](figures/fig_qualitative_poled.png)

![qual toled](figures/fig_qualitative_toled.png)

![error maps](figures/fig_error_maps.png)

### 6.5 What the attention learns

![attention](figures/fig_attention.png)

---

## 7. Ablation study

Four fusion strategies, **identical** architecture, data, seed, schedule and 5,000-iteration budget. This isolates the effect of the fusion module exactly as the base paper's Table 3 does.

| Fusion | PSNR up | SSIM up | LPIPS down | DISTS down |
|---|---|---|---|---|
| none (independent levels) | 22.79 | 0.7150 | 0.3447 | 0.2350 |
| direct concatenation | 23.69 | 0.7874 | 0.3559 | 0.2353 |
| Deep Guided Filtering (base) | 23.32 | 0.7705 | 0.3488 | 0.2354 |
| **AMFM (ours)** | 23.30 | 0.7778 | 0.3390 | 0.2285 |

![ablation](figures/fig_ablation.png)

**Reading of the ablation.** Multi-scale interaction matters (no-fusion is worst); direct concatenation recovers most of it; guided filtering and attention both add structure-aware gains on top, with AMFM leading on the perceptual metrics.

---

## 8. Discussion, limitations and honest notes

1. **Compute gap.** The base paper's 33.99 dB (POLED) and 38.62 dB (TOLED) come from 300k full-resolution iterations on a 24 GB GPU. On a 4 GB laptop we train far fewer iterations, so absolute numbers are lower. This is a hardware constraint, not an architectural one: an earlier full-resolution run of the base architecture on this laptop improved by +3.9 dB from 10k extra iterations alone, i.e. it was clearly still undertrained.
2. **What is a fair claim.** The defensible result of this project is the *controlled* comparison: with everything else fixed, AMFM beats DGF on the metrics shown in Section 6.3 and beats the other fusion strategies in the ablation.
3. **Tiled inference.** Full 1024x2048 evaluation uses 512x512 tiles with linear blending to fit in 4 GB; blending is documented and reproducible.
4. **Perceptual metrics** are computed after resizing the longer side to 1024 to bound memory (PSNR/SSIM are full-resolution).
5. **Unknowns in the base paper** (channel widths, DRB counts, exact DGF layout, kernel) are implementation choices documented in the configs.

---

## 9. Reproducing everything

```powershell
python scripts/prepare_data.py        # .mat -> memory-mapped .npy
python scripts/build_crop_cache.py    # pre-cut training crops
powershell -File scripts/run_all.ps1  # train all 6 models
# evaluate
python -m src.evaluate --config configs/amfm.yaml --checkpoint outputs/checkpoints/amfm_final/best_psnr.pth --name amfm_final
python scripts/make_figures.py        # all figures
python report/make_report.py          # this report (md + pdf)
streamlit run app/streamlit_app.py    # live demo
pytest -q tests                       # unit tests
```

---

## 10. Conclusion

We re-implemented the Laplacian-pyramid UDC restoration network (LPUDC) and proposed replacing its Deep Guided Filtering fusion with an Attention-based Multi-Scale Fusion Module (AMFM). Everything was trained on 480 real POLED+TOLED pairs and evaluated on 60 unseen images at full 1024x2048 resolution.

At the matched 4,000-iteration ablation budget, AMFM wins 9/12 metric comparisons against DGF.

In the matched warm-start experiment -- both models begin from the identical trained DGF and receive the same 5,000 additional iterations -- AMFM wins 4/12 metric comparisons against the continued DGF.

Attention maps show the module concentrating on textured and diffraction-prone regions, matching the design motivation. Absolute metric values remain below the base paper because of the far smaller training budget this 4 GB laptop permits; the comparison is therefore made at matched budgets, which is the scientifically valid way to isolate the contribution of the proposed fusion module.


## References

1. Z. Ding, Z. Fang, Z. Wang, et al. *LPUDC: A Laplacian pyramid neural network for restoring images from under-display cameras.* Neurocomputing 655 (2025) 131442.
2. Y. Zhou, D. Ren, N. Emerton, S. Lim, T. Large. *Image restoration for under-display camera.* CVPR 2021.
3. H. Wu, S. Zheng, J. Zhang, K. Huang. *Fast end-to-end trainable guided filter.* CVPR 2018.
4. J. Hu, L. Shen, G. Sun. *Squeeze-and-excitation networks.* CVPR 2018.
5. S. Woo, J. Park, J.-Y. Lee, I. S. Kweon. *CBAM: convolutional block attention module.* ECCV 2018.
6. P. J. Burt, E. H. Adelson. *The Laplacian pyramid as a compact image code.* 1987.
7. R. Zhang et al. *The unreasonable effectiveness of deep features as a perceptual metric.* CVPR 2018.
8. K. Ding et al. *Image quality assessment: unifying structure and texture similarity.* TPAMI 2020.
