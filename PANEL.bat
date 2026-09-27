@echo off
rem Strata control panel: turn models on and off, see the API address and key, GPU / RAM use.
rem Runs without a window; the "quit" button on the page closes it.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Strata is not installed yet: double-click START-HERE.bat first.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" serve\panel.py --open
