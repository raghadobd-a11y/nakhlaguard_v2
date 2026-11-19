# NakhlaGuard: one-step local start for Windows. Nothing is saved to disk by this script, including the key.
Set-Location -Path $PSScriptRoot
Write-Host "=== NakhlaGuard local start ===" -ForegroundColor Green

# 0) is an older server still running on port 8000?
$busy = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($busy) {
  Write-Host "Port 8000 is busy: an older copy of the server is still running." -ForegroundColor Yellow
  Write-Host "Close the other terminal window (or press Ctrl+C in it), then run this again."
  exit 1
}

# 1) packages (first time only)
python -c "import fastapi, uvicorn, sklearn, httpx, multipart" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "Installing packages (first time only, please wait)..."
  python -m pip install -r requirements.txt
}

# 2) model file (first time only)
if (-not (Test-Path "nakhla_model.joblib")) {
  Write-Host "Training the model (first time only, please wait)..."
  python train_model.py
}

# 3) Gemini key: typed hidden, kept only for this window, never written to a file
if (-not $env:GEMINI_API_KEY) {
  $sec = Read-Host "Paste your Gemini key (hidden) and press Enter. Press Enter alone to skip" -AsSecureString
  $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)
  $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
  [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
  if ($plain) { $env:GEMINI_API_KEY = $plain.Trim() }
}
if ($env:GEMINI_API_KEY) { Write-Host "AI key: set" -ForegroundColor Green } else { Write-Host "AI key: NOT set (the assistant and reports will show a warning)" -ForegroundColor Yellow }

# 4) open the private pages on this laptop only (never use this on Render)
$env:NAKHLA_OWNER_GATE = "off"

# 5) open the browser after 4 seconds, then run the server. This window stays busy while the server runs: that is normal.
Start-Job -ScriptBlock { Start-Sleep -Seconds 4; Start-Process "http://localhost:8000/" } | Out-Null
Write-Host "Starting... the browser opens at http://localhost:8000/   (press Ctrl+C here to stop the server)" -ForegroundColor Green
python app.py
