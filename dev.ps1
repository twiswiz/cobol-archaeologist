# dev.ps1 — start API + web frontend together
# Run from the cobol-archaeologist repo root.
# Pre-requisites (once):
#   pip install uv
#   uv sync
#   cd ..\cobol-archaeologist-web ; npm install
#   ollama pull qwen2.5-coder:1.5b

$apiDir = $PSScriptRoot
$webDir = "$PSScriptRoot\..\cobol-archaeologist-web"

if (-not (Test-Path $webDir)) {
    Write-Error "Web repo not found at '$webDir'. Clone it there first."
    exit 1
}

$api = Start-Process powershell `
    -ArgumentList "-NoExit", "-Command",
        "cd '$apiDir'; uv run uvicorn cobol_archaeologist.api.main:app --reload --port 8000" `
    -PassThru

Write-Host "API started (PID $($api.Id)) -> http://localhost:8000"

$web = Start-Process powershell `
    -ArgumentList "-NoExit", "-Command",
        "cd '$webDir'; npm run dev" `
    -PassThru

Write-Host "Web started (PID $($web.Id)) -> http://localhost:3000"
Write-Host "Close the two new windows (or Ctrl+C in each) to stop."
