# Stage C: full-resolution (512x1024) fine-tuning of the best DGF / AMFM model.
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
Set-Location "D:\UDC_AMFM_Project"
$logs = "outputs\runlogs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
function Run-Step($desc, $cmd) {
    Write-Output "=== $desc ==="
    Invoke-Expression $cmd *> "$logs\$desc.log"
}

# prefer the random-crop retrain as the source when present
$dgf_src  = if (Test-Path "outputs\checkpoints\dgf_cont\best_psnr.pth")  { "dgf_cont"  }
            elseif (Test-Path "outputs\checkpoints\dgf_final\best_psnr.pth") { "dgf_final" } else { "" }
$amfm_src = if (Test-Path "outputs\checkpoints\amfm_warm\best_psnr.pth") { "amfm_warm" }
            elseif (Test-Path "outputs\checkpoints\amfm_final\best_psnr.pth") { "amfm_final" } else { "" }

Remove-Item -Recurse -Force outputs\checkpoints\dgf_ft,outputs\checkpoints\amfm_ft,outputs\logs\dgf_ft,outputs\logs\amfm_ft -ErrorAction SilentlyContinue

if (Test-Path "outputs\checkpoints\$dgf_src\best_psnr.pth") {
    Run-Step "C1_dgf_ft" "python -u -m src.train --config configs/finetune_dgf.yaml --name dgf_ft --init-weights outputs/checkpoints/$dgf_src/best_psnr.pth"
}
if (Test-Path "outputs\checkpoints\$amfm_src\best_psnr.pth") {
    Run-Step "C2_amfm_ft" "python -u -m src.train --config configs/finetune_amfm.yaml --name amfm_ft --init-weights outputs/checkpoints/$amfm_src/best_psnr.pth"
}
Write-Output "FINETUNE DONE (sources: $dgf_src / $amfm_src)"
