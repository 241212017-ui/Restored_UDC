# Full two-stage training + evaluation schedule.
# Stage A : matched 4k-iteration 4-way fusion ablation (none/concat/dgf/amfm)
# Stage B : continue DGF and AMFM to 10k for the headline comparison
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

# ---------------- Stage A : matched ablation (4k iters each) -----------------
Run-Step "A1_none"   "python -u -m src.train --config configs/ablation_none.yaml   --name abl_none"
Run-Step "A2_concat" "python -u -m src.train --config configs/ablation_concat.yaml --name abl_concat"
Run-Step "A3_dgf"    "python -u -m src.train --config configs/dgf.yaml  --name abl_dgf  --iterations 4000"
Run-Step "A4_amfm"   "python -u -m src.train --config configs/amfm.yaml --name abl_amfm --iterations 4000"

# ---------------- Stage B : continue to 10k ----------------------------------
Run-Step "B1_dgf"  "python -u -m src.train --config configs/dgf.yaml  --name dgf_final  --iterations 6000 --lr 0.0001 --init-weights outputs/checkpoints/abl_dgf/best_psnr.pth"
Run-Step "B2_amfm" "python -u -m src.train --config configs/amfm.yaml --name amfm_final --iterations 6000 --lr 0.0001 --init-weights outputs/checkpoints/abl_amfm/best_psnr.pth"

Write-Output "ALL TRAINING DONE"
