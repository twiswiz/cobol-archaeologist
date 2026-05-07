param(
    [switch]$IncludeLarge,
    [switch]$GenerateSynthetic
)

$ErrorActionPreference = "Stop"

$DataRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ManifestPath = Join-Path $DataRoot "manifest.json"
$RawRoot = Join-Path $DataRoot "raw"
$GeneratedRoot = Join-Path $DataRoot "generated"

New-Item -ItemType Directory -Force -Path $RawRoot | Out-Null

$manifest = Get-Content -Raw -Path $ManifestPath | ConvertFrom-Json

foreach ($dataset in $manifest.datasets) {
    if ($dataset.local_generator) {
        Write-Host "Skipping generated dataset '$($dataset.id)' until -GenerateSynthetic is used."
        continue
    }

    if ($dataset.large -and -not $IncludeLarge) {
        Write-Host "Skipping large dataset '$($dataset.id)'. Re-run with -IncludeLarge to download it."
        continue
    }

    $datasetDir = Join-Path $RawRoot $dataset.id
    New-Item -ItemType Directory -Force -Path $datasetDir | Out-Null

    foreach ($file in $dataset.files) {
        $target = Join-Path $datasetDir $file.filename
        if (Test-Path $target) {
            Write-Host "Already present: $target"
            continue
        }

        Write-Host "Downloading $($file.url)"
        try {
            Invoke-WebRequest -Uri $file.url -OutFile $target -ErrorAction Stop
        } catch {
            Write-Warning "Failed to download $($file.url): $($_.Exception.Message)"
            if (Test-Path $target) { Remove-Item $target -Force }
        }
    }
}

if ($GenerateSynthetic) {
    New-Item -ItemType Directory -Force -Path $GeneratedRoot | Out-Null
    $generator = Join-Path $DataRoot "scripts\generate_synthetic_datasets.py"
    python $generator --output-dir $GeneratedRoot
}

Write-Host "Done."
