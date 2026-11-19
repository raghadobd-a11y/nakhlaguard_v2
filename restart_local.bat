@echo off
cd /d "%~dp0"
echo === NakhlaGuard: restart ===

rem 1) stop an older copy of the server that is still listening on port 8000 (only if it is Python)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:":8000 .*LISTENING"') do call :stopit %%a
timeout /t 2 /nobreak >nul

rem 2) start again with the normal start script (it asks for the Gemini key, hidden, never saved)
call start_local.bat
goto :eof

:stopit
tasklist /FI "PID eq %1" /NH 2>nul | findstr /I "python" >nul
if not errorlevel 1 (
  echo Stopping the old server, process %1 ...
  taskkill /F /PID %1 >nul 2>&1
)
goto :eof
