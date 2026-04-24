# Run only the execution step (saved API + programs). Creates program_execution folder when done.
# Uses --stub to skip Molmo (avoids crash). For full accuracy, run without stub when Molmo loads OK.
# Usage: .\run_execution.ps1   or   .\run_execution.ps1 -NumQuestions 10 -NoStub
param([int]$NumQuestions = 5, [switch]$NoStub)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$results  = "results/2026-02-08_19-00-36"
$apiJson  = "$results/api_generator/api.json"
$progJson = "$results/program_generator/programs.json"

if (-not (Test-Path $apiJson))  { Write-Error "Not found: $apiJson" }
if (-not (Test-Path $progJson)) { Write-Error "Not found: $progJson" }

$stubArg = if ($NoStub) { @() } else { @("--stub") }
Write-Host "Running execution for $NumQuestions questions. Results will go to $results/program_execution/"
if (-not $NoStub) { Write-Host "(Using --stub to skip Molmo; answers will be placeholders. Use -NoStub for full run if Molmo loads.)" }
py -3.11 execute_only.py `
  --dataset clevr `
  --annotations-json data/clevr_subset/annotations.json `
  --image-pth data/clevr_subset/images/ `
  --results-pth $results `
  --api-json $apiJson `
  --programs-json $progJson `
  --num-questions $NumQuestions `
  @stubArg
