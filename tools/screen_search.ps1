param(
    [Parameter(Mandatory=$true)][string]$Candidate,
    [Parameter(Mandatory=$true)][string]$Baseline,
    [Parameter(Mandatory=$true)][string]$Output,
    [ValidateRange(1,1000000)][int]$Pairs = 32,
    [string]$TimeControl = '0:01+0.03',
    [ValidateRange(1,8)][int]$Concurrency = 4
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv-nnue/Scripts/python.exe'
$runnerPath = Join-Path $PSScriptRoot 'engine_test.py'
$candidatePath = (Resolve-Path -LiteralPath $Candidate).Path
$baselinePath = (Resolve-Path -LiteralPath $Baseline).Path
$outputPath = [System.IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $outputPath) { throw 'Output must be a new directory' }
New-Item -ItemType Directory -Path $outputPath | Out-Null
$cases = @(
    @{ Name = 'lmr-275'; Option = 'LMR Divisor=275' },
    @{ Name = 'nmp-4'; Option = 'NMP Base Reduction=4' },
    @{ Name = 'rfp-100'; Option = 'RFP Margin=100' },
    @{ Name = 'futility-100'; Option = 'Futility Margin=100' }
)
$cases | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputPath 'candidates.json') -Encoding utf8
$summaries = @()
foreach ($case in $cases) {
    $runPath = Join-Path $outputPath $case.Name
    & $pythonPath $runnerPath $candidatePath $baselinePath $runPath --pairs $Pairs --tc $TimeControl --threads 1 --concurrency $Concurrency --overhead 30 --split calibration --candidate-option $case.Option
    if ($LASTEXITCODE -ne 0) { throw "Invalid experiment: $($case.Name); inspect its log" }
    $summary = Get-Content -LiteralPath (Join-Path $runPath 'summary.json') -Raw | ConvertFrom-Json
    $summaries += @{ Name = $case.Name; Option = $case.Option; Result = $summary }
    $summaries | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $outputPath 'screen.json') -Encoding utf8
}
Write-Output 'Screen complete. These small calibration matches do not confirm a strength gain. Keep defaults until an independent holdout SPRT supports a candidate.'
