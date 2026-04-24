# Run the LEGO spatial pipeline on Windows
# Usage:
#   .\run_lego.ps1                          # LEGO-Lite (default)
#   .\run_lego.ps1 -Full                    # Full benchmark (1100 Qs)
#   .\run_lego.ps1 -MaxQuestions 20         # Small test (20 Qs)
#   .\run_lego.ps1 -MaxQuestions 5 -Stub    # Pipeline test (no models)
param(
    [int]$MaxQuestions = -1,
    [switch]$Lite,
    [switch]$Full,
    [switch]$Stub
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Verify CUDA before starting
Write-Host "Checking CUDA..."
py -3.11 -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA not available! Need NVIDIA GPU.'; print(f'GPU: {torch.cuda.get_device_name(0)}')"
if ($LASTEXITCODE -ne 0) { Write-Error "CUDA check failed. Make sure NVIDIA drivers and CUDA toolkit are installed." }

# Verify Parasail credentials if api.key is absent
if (-not $env:PARASAIL_API_KEY -and -not (Test-Path "api.key")) {
    Write-Error "Set PARASAIL_API_KEY or create api.key before running."
}

# Build arguments
$args_list = @()
if ($MaxQuestions -gt 0) { $args_list += "--max-questions", $MaxQuestions }
if ($Lite)               { $args_list += "--lite" }
if ($Full)               { $args_list += "--full" }
if ($Stub)               { $args_list += "--stub" }

Write-Host "`nRunning the LEGO spatial pipeline..."
Write-Host "Arguments: $($args_list -join ' ')"
Write-Host ""

py -3.11 run_lego.py @args_list
