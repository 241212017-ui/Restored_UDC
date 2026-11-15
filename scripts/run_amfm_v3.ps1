# Retrain the AMFM variant with the attention-augmented guided-fusion design.
# DGF (abl_dgf / dgf_final) is unchanged, so the comparison stays matched.
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

Remove-Item -Recurse -Force outputs\checkpoints\abl_amfm,outputs\checkpoints\amfm_final,outputs\logs\abl_amfm,outputs\logs\amfm_final,outputs\checkpoints\amfm_rc,outputs\logs\amfm_rc,outputs\checkpoints\dgf_rc,outputs\logs\dgf_rc -ErrorAction SilentlyContinue

Run-Step "V1_abl_amfm"  "python -u -m src.train --config configs/amfm.yaml --name abl_amfm --iterations 4000"
Run-Step "V2_amfm_final" "python -u -m src.train --config configs/amfm.yaml --name amfm_final --iterations 6000 --lr 0.0001 --init-weights outputs/checkpoints/abl_amfm/best_psnr.pth"

Write-Output "AMFM V3 DONE"
