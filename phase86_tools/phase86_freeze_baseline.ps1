# Phase 86: archive the already-verified ESP32-C3 Paris demonstration.
# This script never runs inference, builds, or flashes the board.
$ErrorActionPreference = 'Stop'
$root = 'C:\Projects\ESP32-C3-LLM\esp32_c3_llm'
Set-Location $root

$required = @(
  'main\esp32_c3_llm.c',
  'phase85_paris_prompt.py',
  'phase85_paris_prompt_report.json',
  'phase73_instruct\phase80_int8_images\qwen25_instruct_w8_manifest.json'
)
foreach ($rel in $required) {
  if (-not (Test-Path -LiteralPath (Join-Path $root $rel) -PathType Leaf)) {
    throw "Missing required file: $rel. Do not start another inference run; first locate the file."
  }
}

$report = Get-Content -LiteralPath (Join-Path $root 'phase85_paris_prompt_report.json') -Raw | ConvertFrom-Json
$expectedIds = @(63593,315,9625,25)
$actualIds = @($report.prompt_token_ids)
if ($report.prompt -cne 'Capital of France:' -or
    $actualIds.Count -ne 4 -or
    (($actualIds -join ',') -ne ($expectedIds -join ',')) -or
    [int]$report.generated_token -ne 12095 -or
    [bool]$report.target_paris -ne $true -or
    $report.full_text -cne 'Capital of France: Paris') {
  throw 'The local report does not match the verified Paris run. Archive aborted.'
}
$checks = @($report.layer1_checks)
if ($checks.Count -ne 4 -or (@($checks | Where-Object { [int]$_.failures -ne 0 }).Count -ne 0)) {
  throw 'Local report does not contain four passing layer-1 checks. Archive aborted.'
}

$target = Join-Path $root 'phase86_verified_paris_baseline'
if (Test-Path -LiteralPath $target) {
  throw "Archive folder already exists: $target. Rename it or inspect it before rerunning."
}
New-Item -ItemType Directory -Path (Join-Path $target 'main') -Force | Out-Null

$toCopy = @(
  'main\esp32_c3_llm.c', 'phase85_paris_prompt.py', 'phase85_paris_prompt_report.json',
  'phase73_instruct\phase80_int8_images\qwen25_instruct_w8_manifest.json'
)
foreach ($optional in @('sdkconfig', 'CMakeLists.txt','main\CMakeLists.txt','partitions.csv')) {
  if (Test-Path -LiteralPath (Join-Path $root $optional) -PathType Leaf) { $toCopy += $optional }
}
foreach ($rel in $toCopy) {
  $dest = Join-Path $target $rel
  New-Item -ItemType Directory -Path (Split-Path $dest -Parent) -Force | Out-Null
  Copy-Item -LiteralPath (Join-Path $root $rel) -Destination $dest -Force
}
$hashList = foreach ($rel in $toCopy) {
  $file = Join-Path $target $rel
  [pscustomobject]@{ file=$rel; sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $file).Hash.ToLowerInvariant() }
}
$hashList | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $target 'source_sha256.json') -Encoding UTF8

$description = @'
# Phase 86: Verified PC-assisted ESP32-C3 LLM demonstration

Prompt: `Capital of France:`
Tokenizer IDs: `[63593, 315, 9625, 25]`
ESP32-generated next token: `12095` (` Paris`)
Decoded output: `Capital of France: Paris`
Device result: `PHASE85 DEVICE GATE PASS prompt_tokens=4 generated_tokens=1 layers=24`

All 24 transformer layers, KV cache/attention and output-head token selection ran on the device. PC served embeddings/weights and performed offline validation. Full layer-by-layer independent reference validation was NOT performed; first-layer checks at four positions and the output-head audit passed.

This archive contains source, a JSON run report, checksums, configuration (if found), and the reference model export manifest. It intentionally DOES NOT contain massive weight images, virtual environments, builds, or credentials. Original exported model files must remain in the project's `phase73_instruct/phase80_int8_images` directory for reproduction.

No firmware rebuild, flash, USB connection, or additional inference run is needed to create this archive.
'@
Set-Content -LiteralPath (Join-Path $target 'README.md') -Value $description -Encoding UTF8
$zip = Join-Path $root 'phase86_verified_paris_baseline.zip'
if (Test-Path -LiteralPath $zip) { throw "ZIP already exists: $zip; archive folder created, but existing ZIP was not replaced." }
Compress-Archive -Path (Join-Path $target '*') -DestinationPath $zip -CompressionLevel Optimal
Write-Host 'PHASE86 ARCHIVE PASS'
Write-Host "Folder: $target"
Write-Host "ZIP: $zip"
Write-Host ('ZIP SHA256: ' + (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash)
