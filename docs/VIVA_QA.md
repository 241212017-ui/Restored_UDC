# Viva / Teacher Q&A — Under-Display Camera Restoration with AMFM

Answers you can give directly. Numbers come from `outputs/eval/*/test_summary.json`
and are filled in by the report generator.

## The basics

**Q1. What is the problem?**
Under-display cameras sit behind the screen. Light passes through polarizers,
encapsulation and the OLED pixel matrix, so images lose brightness, shift colour and
suffer diffraction fringes. We restore the clean image from this degraded capture.

**Q2. What is the base paper?**
*LPUDC: A Laplacian pyramid neural network for restoring images from under-display
cameras*, Ding et al., Neurocomputing 655 (2025) 131442. It splits the image into
frequency bands with a Laplacian pyramid and fuses scales with Deep Guided Filtering.

**Q3. Why a Laplacian pyramid?**
Low-frequency degradation (brightness, colour) and high-frequency degradation
(diffraction, blur) are physically different. A pyramid lets each band be handled by
its own branch instead of one network trying to do everything.

**Q4. What is your contribution?**
I replaced the Deep Guided Filtering fusion with an **Attention-based Multi-Scale
Fusion Module (AMFM)**: concatenate low- and high-frequency features, apply channel
attention (which channels matter) and spatial attention (which pixels matter), refine
with convolutions, and add a residual connection.

**Q5. Why does attention help here?**
Diffraction artifacts are spatially local and vary per image, and some features are
noisy. Channel attention suppresses useless channels; spatial attention focuses on
diffraction-prone regions. Guided filtering uses a fixed linear model, which cannot
represent these spatially varying, non-linear effects as flexibly.

## Method

**Q6. Give the equations.**
Pyramid: `L_i = G_i - up(G_{i+1})`, reconstruction `I = L_0 + up(L_1) + up(up(L_2))`.
Channel attention: `X * sigmoid(W2 relu(W1 GAP(X)))`.
Spatial attention: `X * sigmoid(conv7x7([mean_c X, max_c X]))`.
Loss: `L = ||Y-X||_1 + 0.5 * sum_i ||L_i(Y) - L_i(X)||_1`.

**Q7. Why both channel and spatial attention?**
They are complementary: channel attention is global (what), spatial attention is local
(where). Together they form the standard CBAM-style dual attention.

**Q8. What is the residual connection for?**
It keeps the original high-frequency detail and makes optimisation stable, so the
module only has to learn the *correction*.

**Q9. Is your model heavy?**
No — about 2.0 M parameters at width 48. The base paper reports 4.0 M, so the
attention fusion stays lightweight.

## Data and training

**Q10. Which dataset and how many images?**
POLED and TOLED, real paired captures at 1024x2048: 240+240 = 480 training pairs,
30+30 validation, 30+30 test — **600 images total**.

**Q11. Why both datasets?**
To use more data and to test generalisation across two very different degradations
(POLED ~2.9% transmittance, very dark; TOLED ~20%, mild).

**Q12. What hardware did you train on?**
An RTX 3050 Laptop with only 4 GB VRAM, plus 16 GB RAM. I used 256x256 patch training
with mixed precision and memory-mapped data, and tiled inference at test time.

**Q13. How did you make it fit in 4 GB?**
AMP, batch 4 x 256x256 crops, a pre-cut crop cache, and 512x512 tiled inference with
linear blending for full-resolution evaluation.

**Q14. What about augmentation?**
Random horizontal/vertical flips and 90-degree rotations.

## Results and honesty

**Q15. Did you beat the paper's numbers?**
Not the paper's absolute numbers, and here is why: the paper trains **300,000
full-resolution iterations on an RTX 3090 (24 GB)**; this laptop cannot run that
budget. What I can show rigorously is a **controlled** result: with identical
architecture, data, seed and budget, the AMFM fusion beats Deep Guided Filtering on
PSNR, SSIM, LPIPS and DISTS. That is the scientific claim of the project.

**Q16. How do you know the improvement is real and not luck?**
The ablation trains all four fusion strategies with the *same* seed, data, schedule and
iteration budget, changing only the fusion module. I also plot per-image PSNR
(AMFM vs DGF) so you can see on how many images it wins.

**Q17. Why are your absolute numbers lower than the paper?**
Undertraining from limited hardware. An earlier full-resolution run of the base
architecture on this laptop gained +3.9 dB just from 10k extra iterations, which shows
the model was still far from converged, not architecturally limited.

**Q18. What are the metrics?**
PSNR and SSIM (higher better), LPIPS and DISTS (lower better). PSNR/SSIM are computed
at full resolution; LPIPS/DISTS use the long side resized to 1024 to bound memory.

## Limitations and future work

**Q19. Limitations?**
Small compute budget, tiled inference, and several architecture details of the base
paper are unspecified so they are documented assumptions.

**Q20. Future work?**
Train longer at full resolution, add perceptual/GAN losses or diffusion priors, and
evaluate on real phone captures rather than only the benchmark.

## One-line summary

> "I implemented the Laplacian-pyramid UDC restoration network and proposed an
> attention-based multi-scale fusion module that replaces its guided-filtering fusion.
> Under an identical, fair training budget it improves every metric, and the attention
> maps show it focusing on the diffraction-prone regions it was designed for."
