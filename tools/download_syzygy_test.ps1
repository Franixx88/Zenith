param([string]$Destination = (Join-Path $PSScriptRoot '../target/syzygy-tests'))
$ErrorActionPreference = 'Stop'
$targetPath = [IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $targetPath | Out-Null
$manifest = @()
$expected = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'syzygy-test-manifest.json') -Raw | ConvertFrom-Json
foreach ($material in @('KQvK', 'KRvK', 'KBvK', 'KNvK', 'KPvK')) {
    foreach ($kind in @('wdl', 'dtz')) {
        $extension = if ($kind -eq 'wdl') { 'rtbw' } else { 'rtbz' }
        $name = "$material.$extension"
        $url = "https://tablebase.lichess.ovh/tables/standard/3-4-5-$kind/$name"
        $file = Join-Path $targetPath $name
        if (-not (Test-Path -LiteralPath $file)) {
            Invoke-WebRequest -Uri $url -OutFile $file -UseBasicParsing
        }
        $digest = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash
        $reference = $expected | Where-Object { $_.file -eq $name }
        if ($digest -ne $reference.sha256) { throw "SHA256 mismatch for $file" }
        $manifest += [pscustomobject]@{file=$name; bytes=(Get-Item -LiteralPath $file).Length; sha256=$digest; source=$url}
    }
}
$manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $targetPath 'manifest.json') -Encoding utf8
$manifest | Format-Table file, bytes, sha256
