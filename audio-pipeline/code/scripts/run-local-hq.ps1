[CmdletBinding()]
param(
    [double]$Minutes = 0,
    [ValidateSet("lofi", "jazz", "rnb")]
    [string]$Genre = "lofi",
    [ValidateSet("random", "relaxed", "melancholic", "uplifting", "dreamy", "nostalgic", "cozy")]
    [string]$Mood = "random",
    [string]$Quality = "",
    [double]$Bpm = 0,
    [double]$RenderBudgetHours = 6,
    [string]$ModelPool = "facebook/musicgen-large",
    [string]$OutWav = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$scriptPath = Join-Path $repoRoot "archiv\hybrid_ap4\ap4_prompting_musikgenerierung\foundation_prompting.py"
$featuresPath = Join-Path $repoRoot "daten\features\audio_features.jsonl"

function Read-DoubleInRange {
    param(
        [string]$Prompt,
        [double]$Default,
        [double]$Min,
        [double]$Max
    )

    while ($true) {
        $raw = Read-Host "$Prompt [$Default]"
        if (-not $raw.Trim()) {
            return $Default
        }

        $candidate = 0.0
        if ([double]::TryParse($raw.Replace(",", "."), [ref]$candidate) -and $candidate -ge $Min -and $candidate -le $Max) {
            return $candidate
        }

        Write-Host "Bitte eine Zahl zwischen $Min und $Max eingeben." -ForegroundColor Yellow
    }
}

function Read-Choice {
    param(
        [string]$Prompt,
        [string[]]$Choices,
        [string]$Default
    )

    $choiceLabel = ($Choices -join "/")
    while ($true) {
        $raw = Read-Host "$Prompt ($choiceLabel) [$Default]"
        if (-not $raw.Trim()) {
            return $Default
        }

        $normalized = $raw.Trim().ToLower()
        if ($Choices -contains $normalized) {
            return $normalized
        }

        Write-Host "Bitte einen der folgenden Werte eingeben: $choiceLabel" -ForegroundColor Yellow
    }
}

if (-not (Test-Path $scriptPath)) {
    throw "foundation_prompting.py nicht gefunden: $scriptPath"
}

$missingPkgs = & python -c "import importlib.util,sys; mods=['torch','torchaudio','audiocraft']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; print(','.join(missing)); sys.exit(1 if missing else 0)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[error] Fehlende Python-Pakete: $missingPkgs"
    Write-Host "Installiere sie mit: pip install -r backend\requirements.txt"
    exit 1
}

if ($Minutes -le 0) {
    $Minutes = Read-DoubleInRange -Prompt "Wie lange soll die Audio gehen? (1-2 Minuten empfohlen)" -Default 1 -Min 1 -Max 2
}
if (-not $Quality.Trim()) {
    $Quality = Read-Choice -Prompt "Welche Qualitaet soll verwendet werden?" -Choices @("auto", "fast", "medium", "standard", "ultra", "premium", "final_master") -Default "final_master"
}
if ($Bpm -le 0) {
    $Bpm = Read-DoubleInRange -Prompt "Wie viel BPM soll die Audio haben?" -Default 90 -Min 60 -Max 140
}

$args = @(
    $scriptPath,
    "--minutes", "$Minutes",
    "--genre", $Genre,
    "--mood", $Mood,
    "--quality", $Quality,
    "--bpm", "$Bpm",
    "--log-level", "INFO"
)

if (Test-Path $featuresPath) {
    $args += @("--features", $featuresPath)
}

if ($RenderBudgetHours -gt 0) {
    $args += @("--render-budget-hours", "$RenderBudgetHours")
}

if ($ModelPool.Trim()) {
    $args += @("--model-pool", $ModelPool.Trim())
}

if ($OutWav.Trim()) {
    $args += @("--out-wav", $OutWav.Trim())
}

Write-Host "[run] python $($args -join ' ')"
python @args
exit $LASTEXITCODE
