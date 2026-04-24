# Download LEGO Benchmark data (CLEVR subset from Google Drive). Run from project root.
# Omni3D-Bench: get from https://huggingface.co/datasets/dmarsili/Omni3D-Bench and place omni3d-bench.zip in data/ if needed.

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = (Get-Location).Path }

Push-Location $ProjectRoot

if (-not (Test-Path "data")) { New-Item -ItemType Directory -Path "data" | Out-Null }
Set-Location "data"

Write-Host "Installing gdown..."
py -3.11 -m pip install gdown --quiet

Write-Host "Downloading clevr_subset.zip from Google Drive..."
py -3.11 -m gdown "https://drive.google.com/uc?id=1gWxr19YjQQAvgIUQSsw2Kr4w-q-F0pgx" -O clevr_subset.zip

if (Test-Path "clevr_subset.zip") {
    Write-Host "Extracting clevr_subset.zip..."
    Expand-Archive -Path "clevr_subset.zip" -DestinationPath "." -Force
    Write-Host "Done. CLEVR subset is in data/clevr_subset/"
} else {
    Write-Host "Download failed. Try running in Git Bash: sh download_data.sh"
}

# Optional: if omni3d-bench.zip is in data/, unzip it
if (Test-Path "omni3d-bench.zip") {
    Write-Host "Extracting omni3d-bench.zip..."
    Expand-Archive -Path "omni3d-bench.zip" -DestinationPath "omni3d-bench" -Force
}

Pop-Location
