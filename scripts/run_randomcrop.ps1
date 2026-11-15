# Random-crop retrain: removes the fixed-crop-pool confound.
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
Run-Step "R1_dgf_rc"  "python -u -m src.train --config configs/randomcrop_dgf.yaml  --name dgf_rc"
Run-Step "R2_amfm_rc" "python -u -m src.train --config configs/randomcrop_amfm.yaml --name amfm_rc"
Write-Output "RANDOMCROP DONE"
