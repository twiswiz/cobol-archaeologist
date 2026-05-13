# dev.ps1 — start API + web frontend together
# Run once first: pip install -e ".[api,rag]"  and  npm install (in the web dir)

$apiDir = $PSScriptRoot
$webDir = "$PSScriptRoot\..\cobol-archaeologist-web"

$api = Start-Process powershell `
  -ArgumentList "-NoExit", "-Command", "cd '$apiDir'; uvicorn cobol_archaeologist.api.main:app --reload --port 8000" `
  -PassThru
Write-Host "API started (PID $($api.Id)) -> http://localhost:8000"

$web = Start-Process powershell `
  -ArgumentList "-NoExit", "-Command", "cd '$webDir'; npm run dev" `
  -PassThru
Write-Host "Web started (PID $($web.Id)) -> http://localhost:3000"
Write-Host "Close the two new windows (or Ctrl+C in each) to stop."
