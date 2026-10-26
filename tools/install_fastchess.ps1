$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$targetPath = Join-Path $projectRoot 'target'
$archivePath = Join-Path $targetPath 'fastchess.zip'
$destinationPath = Join-Path $targetPath 'fastchess'
$expectedSha = '007ce550dc809510b34622a1aeb43fd443d40284045bef5c1bc39c8941114143'
$releaseUrl = 'https://github.com/Disservin/fastchess/releases/download/v1.8.2-alpha/fastchess-windows-x86-64.zip'
New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
if (-not (Test-Path -LiteralPath $archivePath)) {
    Invoke-WebRequest -Uri $releaseUrl -OutFile $archivePath
}
if ((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedSha) {
    throw 'Fastchess archive SHA-256 mismatch; refusing to extract or run it.'
}
if (-not (Test-Path -LiteralPath $destinationPath)) {
    Expand-Archive -LiteralPath $archivePath -DestinationPath $destinationPath
}
$binaryPath = Join-Path $destinationPath 'fastchess-windows-x86-64/fastchess.exe'
if (-not (Test-Path -LiteralPath $binaryPath)) { throw 'Missing Fastchess executable' }
if ((Get-FileHash -LiteralPath $binaryPath -Algorithm SHA256).Hash -ne 'C6D7E4458A58025983C81BF38C524CB213E6F27559657DCFD72543392A431A71') {
    throw 'Installed Fastchess executable SHA-256 mismatch.'
}
Write-Output $binaryPath
