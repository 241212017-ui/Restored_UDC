# Final chain: retrain the 4k AMFM ablation point with the current module,
# fine-tune both headline arms, evaluate everything, then rebuild
# figures / report / slides and run the tests.
$ErrorActionPreference = "Continue"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
Set-Location "D:\UDC_AMFM_Project"
$logs = "outputs\runlogs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
function Run-Step($desc, $cmd) {
    Write-Output "=== $desc ==="
    $t0 = Get-Date
    Invoke-Expression $cmd *> "$logs\$desc.log"
    Write-Output "  done in $([math]::Round(((Get-Date)-$t0).TotalMinutes,1)) min"
}

# 1) matched 4k ablation point for the CURRENT AMFM design
Run-Step "F1_abl_amfm" "python -u -m src.train --config configs/amfm.yaml --name abl_amfm --iterations 4000"

# 2) full-resolution fine-tune of both matched arms
Run-Step "F2_dgf_ft"  "python -u -m src.train --config configs/finetune_dgf.yaml  --name dgf_ft  --init-weights outputs/checkpoints/dgf_cont/best_psnr.pth"
Run-Step "F3_amfm_ft" "python -u -m src.train --config configs/finetune_amfm.yaml --name amfm_ft --init-weights outputs/checkpoints/amfm_warm/best_psnr.pth"

# 3) evaluation of everything
Run-Step "F4_eval" "powershell -ExecutionPolicy Bypass -File scripts/run_eval.ps1"

# 4) deliverables
Run-Step "F5_figures" "python scripts/make_figures.py"
Run-Step "F6_report"  "python report/make_report.py"
Run-Step "F7_slides"  "python slides/make_slides.py"
Run-Step "F8_tests"   "python -m pytest -q tests"

Write-Output "FINAL RUN DONE"
