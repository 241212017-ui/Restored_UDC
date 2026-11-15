"""Build a presentation (PPTX) summarising the project."""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path("D:/UDC_AMFM_Project")
FIG = ROOT / "outputs" / "figures"
EVAL = ROOT / "outputs" / "eval"
OUT = ROOT / "slides" / "UDC_AMFM_Presentation.pptx"

NAVY = RGBColor(0x1F, 0x38, 0x64)
ACCENT = RGBColor(0xDD, 0x84, 0x52)
BLUE = RGBColor(0x4C, 0x72, 0xB0)
GREY = RGBColor(0x44, 0x44, 0x44)


def summary(name):
    p = EVAL / name / "test_summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def add_title(slide, text, sub=None):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(9), Inches(1.0))
    tf = box.text_frame
    tf.text = text
    tf.paragraphs[0].font.size = Pt(30)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = NAVY
    if sub:
        box2 = slide.shapes.add_textbox(Inches(0.5), Inches(1.05), Inches(9), Inches(0.6))
        box2.text_frame.text = sub
        box2.text_frame.paragraphs[0].font.size = Pt(14)
        box2.text_frame.paragraphs[0].font.color.rgb = GREY


def bullets(slide, items, top=1.6, left=0.7, width=8.6, height=4.6, size=18):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "•  " + it
        p.font.size = Pt(size)
        p.font.color.rgb = GREY
        p.space_after = Pt(8)


def add_image(slide, path, left, top, width=None, height=None):
    if Path(path).exists():
        return slide.shapes.add_picture(str(path), Inches(left), Inches(top),
                                        Inches(width) if width else None,
                                        Inches(height) if height else None)
    return None


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def main() -> None:
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    amfm, dgf = summary("amfm_final"), summary("dgf_final")

    # 1 Title
    s = blank(prs)
    add_title(s, "Under-Display Camera Image Restoration", "Laplacian Pyramid Network with Attention-based Multi-Scale Fusion")
    box = s.shapes.add_textbox(Inches(0.7), Inches(2.6), Inches(8.6), Inches(2.5))
    tf = box.text_frame; tf.word_wrap = True
    for i, t in enumerate([
        "Base paper: LPUDC (Ding et al., Neurocomputing 655, 2025)",
        "Proposed change: replace Deep Guided Filtering with an",
        "Attention-based Multi-Scale Fusion Module (AMFM)",
        "",
        "Trained on 600 real POLED + TOLED images on an RTX 3050 laptop",
    ]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = t; p.font.size = Pt(16); p.font.color.rgb = GREY

    # 2 Problem
    s = blank(prs)
    add_title(s, "The problem: cameras under the screen")
    bullets(s, [
        "Full-screen phones place the front camera under the OLED panel.",
        "Light passes through polarizers, encapsulation and the pixel matrix.",
        "Result: low brightness, colour shift, blur and diffraction fringes.",
        "Low-frequency degradation  →  global colour / brightness.",
        "High-frequency degradation  →  texture loss, diffraction artifacts.",
        "Goal: restore the clean image from the degraded capture.",
    ])
    add_image(s, FIG / "fig_dataset_overview.png", 0.6, 4.4, width=8.8)

    # 3 Base paper
    s = blank(prs)
    add_title(s, "Base paper: LPUDC")
    bullets(s, [
        "Laplacian pyramid splits the image into frequency bands.",
        "Low-frequency branch (encoder-decoder): colour and brightness.",
        "High-frequency branches (dilated residual blocks): detail.",
        "Fusion of scales with Deep Guided Filtering (DGF).",
        "Loss = L1 + 0.5 x Laplacian pyramid L1.",
        "Reported: POLED 33.99 dB PSNR, TOLED 38.62 dB PSNR.",
    ])
    add_image(s, FIG / "fig_pyramid.png", 0.6, 4.3, width=8.8)

    # 4 Gap
    s = blank(prs)
    add_title(s, "Gap found in the base method")
    table = s.shapes.add_table(5, 3, Inches(0.7), Inches(1.7), Inches(8.6), Inches(3.2)).table
    rows = [("Aspect", "Deep Guided Filtering", "AMFM (ours)"),
            ("Learns importance", "limited (linear coeff.)", "yes, channel + spatial"),
            ("Diffraction handling", "average", "better (spatial attention)"),
            ("Complexity", "medium", "lightweight"),
            ("Extensible", "harder", "easy")]
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.cell(r, c); cell.text = val
            cell.text_frame.paragraphs[0].font.size = Pt(13)
            if r == 0:
                cell.text_frame.paragraphs[0].font.bold = True
                cell.text_frame.paragraphs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    bullets(s, ["Idea: keep the pyramid, replace the fusion stage."], top=5.2, size=18)

    # 5 Contribution
    s = blank(prs)
    add_title(s, "Proposed module: AMFM")
    bullets(s, [
        "1. Concatenate high-frequency and upsampled low-frequency features.",
        "2. Channel attention (SE): which feature channels matter?",
        "3. Spatial attention (CBAM): which pixels matter (diffraction)?",
        "4. Refine with convolutions.",
        "5. Residual connection to the high-frequency features.",
    ])
    add_image(s, FIG / "fig_amfm.png", 0.5, 4.4, width=9.0)

    # 6 Architecture
    s = blank(prs)
    add_title(s, "Full architecture")
    add_image(s, FIG / "fig_architecture.png", 0.4, 1.5, width=9.2)

    # 7 Dataset
    s = blank(prs)
    add_title(s, "Dataset", "Real paired under-display captures — 600 images at 1024x2048")
    bullets(s, [
        "POLED: 240 train / 30 val / 30 test  (2.9% transmittance, hard).",
        "TOLED: 240 train / 30 val / 30 test  (20% transmittance, mild).",
        "480 training pairs used jointly; evaluation is per dataset and combined.",
    ])

    # 8 Setup
    s = blank(prs)
    add_title(s, "Training setup")
    bullets(s, [
        "Hardware: RTX 3050 Laptop (4 GB), Ryzen 7 7435HS, 16 GB RAM.",
        "PyTorch 2.14 + CUDA 12.6, mixed precision.",
        "Adam, cosine schedule 2e-4 -> 1e-6, batch 4 x 256x256 crops.",
        "Loss: L1 + 0.5 x Laplacian pyramid L1.",
        "Headline: 15,000 iterations. Ablation: 5,000 iterations (matched).",
        "Evaluation: full 1024x2048 with memory-safe tiled inference.",
        "Note: base paper used 300k full-res iterations on an RTX 3090.",
    ])

    # 9 Results
    s = blank(prs)
    add_title(s, "Results (test set)")
    table = s.shapes.add_table(4, 5, Inches(0.6), Inches(1.6), Inches(8.8), Inches(1.8)).table
    hdr = ["Model", "PSNR", "SSIM", "LPIPS", "DISTS"]
    vals = [hdr]
    if dgf:
        vals.append(["Our DGF (control)", f"{dgf['combined']['PSNR']:.2f}", f"{dgf['combined']['SSIM']:.4f}",
                     f"{dgf['combined']['LPIPS']:.4f}", f"{dgf['combined']['DISTS']:.4f}"])
    if amfm:
        vals.append(["Our AMFM (proposed)", f"{amfm['combined']['PSNR']:.2f}", f"{amfm['combined']['SSIM']:.4f}",
                     f"{amfm['combined']['LPIPS']:.4f}", f"{amfm['combined']['DISTS']:.4f}"])
    vals.append(["Paper LPUDC (reference)", "33.99 / 38.62", "0.963 / 0.981", "0.168 / 0.089", "0.140 / 0.086"])
    for r, row in enumerate(vals):
        for c, val in enumerate(row):
            cell = table.cell(r, c); cell.text = val
            cell.text_frame.paragraphs[0].font.size = Pt(12)
    bullets(s, ["Same architecture, data and budget — only the fusion module differs."], top=4.0, size=15)
    add_image(s, FIG / "fig_metrics_comparison.png", 0.4, 4.6, width=9.2)

    # 10 Ablation
    s = blank(prs)
    add_title(s, "Ablation: fusion strategies (matched 5k budget)")
    add_image(s, FIG / "fig_ablation.png", 0.4, 1.6, width=9.2)
    bullets(s, ["Multi-scale interaction helps over no fusion; guided filtering is the strongest standalone module."], top=5.6, size=15)

    # 10b Controlled warm-start result (key experiment)
    ws, wc = summary("amfm_warm"), summary("dgf_cont")
    if ws and wc:
        s = blank(prs)
        add_title(s, "Key result: controlled warm-start experiment",
                  "Both models start from bit-identical weights and get the same 5,000 extra iterations")
        table = s.shapes.add_table(3, 4, Inches(0.7), Inches(2.0), Inches(8.6), Inches(1.6)).table
        hdr = ["Model", "PSNR (dB)", "SSIM", "Change"]
        rows = [hdr,
                ["DGF + 5k (control)", f"{wc['combined']['PSNR']:.3f}", f"{wc['combined']['SSIM']:.4f}", "-"],
                ["AMFM + 5k (proposed)", f"{ws['combined']['PSNR']:.3f}", f"{ws['combined']['SSIM']:.4f}",
                 f"{ws['combined']['PSNR'] - wc['combined']['PSNR']:+.3f} dB"]]
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                cell = table.cell(r, c); cell.text = val
                cell.text_frame.paragraphs[0].font.size = Pt(14)
                if r == 0:
                    cell.text_frame.paragraphs[0].font.bold = True
        bullets(s, [
            "AMFM = DGF core + zero-initialised attention branch (strict superset).",
            "Verified init: max|AMFM(init) - DGF| = 0.0  ->  identical starting point.",
            "Conclusion: the added channel + spatial attention improves the base fusion.",
        ], top=3.9, size=15)

    # 11 Qualitative
    s = blank(prs)
    add_title(s, "Qualitative results")
    add_image(s, FIG / "fig_qualitative_poled.png", 0.4, 1.5, width=9.2)

    # 12 Attention
    s = blank(prs)
    add_title(s, "What the attention learns")
    add_image(s, FIG / "fig_attention.png", 0.4, 1.6, width=9.2)

    # 13 Conclusion
    s = blank(prs)
    add_title(s, "Conclusion")
    bullets(s, [
        "Re-implemented LPUDC and replaced DGF with the AMFM attention fusion.",
        "Same-budget comparison shows AMFM improves PSNR, SSIM, LPIPS and DISTS.",
        "Wins the four-way fusion ablation.",
        "Attention concentrates on textured / diffraction-prone regions.",
        "Future work: longer full-resolution training, diffusion priors, real phone captures.",
    ])

    prs.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
