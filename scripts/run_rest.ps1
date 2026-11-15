# Resume schedule: re-run DGF + AMFM ablation (fixed residual head) then stage B.
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

# remove the stale (pre-fix) checkpoints/logs
Remove-Item -Recurse -Force outputs\checkpoints\abl_dgf,outputs\checkpoints\abl_amfm,outputs\logs\abl_dgf,outputs\logs\abl_amfm -ErrorAction SilentlyContinue

Run-Step "A3_dgf"  "python -u -m src.train --config configs/dgf.yaml  --name abl_dgf  --iterations 4000"
Run-Step "A4_amfm" "python -u -m src.train --config configs/amfm.yaml --name abl_amfm --iterations 4000"
Run-Step "B1_dgf"  "python -u -m src.train --config configs/dgf.yaml  --name dgf_final  --iterations 6000 --lr 0.0001 --init-weights outputs/checkpoints/abl_dgf/best_psnr.pth"
Run-Step "B2_amfm" "python -u -m src.train --config configs/amfm.yaml --name amfm_final --iterations 6000 --lr 0.0001 --init-weights outputs/checkpoints/abl_amfm/best_psnr.pth"

Write-Output "RESUME DONE"
