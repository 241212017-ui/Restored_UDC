# Finalize: retrain the 4k AMFM ablation point with the current module, then
# evaluate everything and rebuild all deliverables.  The full-resolution
# fine-tune stage is intentionally omitted: on this laptop its PNG data pipeline
# runs at ~3.5 s/iteration, which is not affordable.
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

Run-Step "G1_abl_amfm" "python -u -m src.train --config configs/amfm.yaml --name abl_amfm --iterations 4000"
Run-Step "G2_eval"     "powershell -ExecutionPolicy Bypass -File scripts/run_eval.ps1"
Run-Step "G3_figures"  "python scripts/make_figures.py"
Run-Step "G4_report"   "python report/make_report.py"
Run-Step "G5_slides"   "python slides/make_slides.py"
Run-Step "G6_tests"    "python -m pytest -q tests"

Write-Output "FINALIZE DONE"
