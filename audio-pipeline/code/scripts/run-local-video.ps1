[CmdletBinding()]
param(
    [ValidateSet("draft", "standard", "premium", "final_master")]
    [string]$Quality = "premium",
    [double]$Duration = 60,
    [string]$Mood = "",
    [string]$Setting = "",
    [string]$Character = "",
    [string]$Atmosphere = "",
    [string]$TimeOfDay = "",
    [string]$Lighting = "",
    [string]$RenderStyle = "",
    [string]$Audio = "",
    [int]$Seed = -1,
    [string]$OutputDir = "",
    [string]$BaseName = "",
    [string]$LogLevel = "INFO"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$scriptPath = Join-Path $repoRoot "src\Video\local_video.py"

if (-not (Test-Path $scriptPath)) {
    throw "local_video.py nicht gefunden: $scriptPath"
}

$missingPkgs = & python -c "import importlib.util,sys; mods=['torch','diffusers','transformers','accelerate','safetensors','PIL']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; print(','.join(missing)); sys.exit(1 if missing else 0)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[error] Fehlende Python-Pakete: $missingPkgs"
    Write-Host "Installiere sie mit: pip install -r backend\requirements.txt"
    exit 1
}

if (-not $OutputDir.Trim()) {
    $OutputDir = Join-Path $repoRoot "outputs\generated_video"
}

$pyArgs = @(
    $scriptPath,
    "--quality", $Quality,
    "--duration", "$Duration",
    "--output-dir", $OutputDir,
    "--log-level", $LogLevel
)

if ($Mood.Trim())        { $pyArgs += @("--mood", $Mood.Trim()) }
if ($Setting.Trim())     { $pyArgs += @("--setting", $Setting.Trim()) }
if ($Character.Trim())   { $pyArgs += @("--character", $Character.Trim()) }
if ($Atmosphere.Trim())  { $pyArgs += @("--atmosphere", $Atmosphere.Trim()) }
if ($TimeOfDay.Trim())   { $pyArgs += @("--time-of-day", $TimeOfDay.Trim()) }
if ($Lighting.Trim())    { $pyArgs += @("--lighting", $Lighting.Trim()) }
if ($RenderStyle.Trim()) { $pyArgs += @("--render-style", $RenderStyle.Trim()) }
if ($Audio.Trim())       { $pyArgs += @("--audio", $Audio.Trim()) }
if ($Seed -ge 0)         { $pyArgs += @("--seed", "$Seed") }
if ($BaseName.Trim())    { $pyArgs += @("--base-name", $BaseName.Trim()) }

Write-Host "[run] python $($pyArgs -join ' ')"
python @pyArgs
exit $LASTEXITCODE
