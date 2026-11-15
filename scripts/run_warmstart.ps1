# Matched warm-start experiment.
#   dgf_cont  : the trained DGF model trained 5k further  (control)
#   amfm_warm : the same trained DGF warm-started into AMFM, trained 5k (proposed)
# Both start from the identical point and get the identical budget, so any
# difference is caused purely by the added attention branch.
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

Remove-Item -Recurse -Force outputs\checkpoints\dgf_cont,outputs\logs\dgf_cont,outputs\checkpoints\amfm_warm,outputs\logs\amfm_warm -ErrorAction SilentlyContinue

Run-Step "W1_dgf_cont"  "python -u -m src.train --config configs/cont_dgf.yaml  --name dgf_cont  --init-weights outputs/checkpoints/dgf_final/best_psnr.pth"
Run-Step "W2_amfm_warm" "python -u -m src.train --config configs/cont_amfm.yaml --name amfm_warm --init-weights outputs/checkpoints/amfm_warm_init/init.pth"

Write-Output "WARMSTART DONE"
