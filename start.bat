@echo off
title CYCLONE-X Launcher
color 0B

echo.
echo  ====================================================
echo    CYCLONE-X  ^|  AI Cyclone Intelligence Platform
echo  ====================================================
echo.

:: --- Resolve project root (directory of this script) ---
set ROOT=%~dp0
cd /d "%ROOT%"

:: --- Resolve Python executable (.venv if present, otherwise global python) ---
set PYTHON_EXE=python
if exist "%ROOT%.venv\Scripts\python.exe" (
    set PYTHON_EXE="%ROOT%.venv\Scripts\python.exe"
)

:: --- 1. Start FastAPI Backend in a new window ---
echo  [1/2] Starting FastAPI backend on http://localhost:8000 ...
start "CYCLONE-X Backend" cmd /k "title CYCLONE-X Backend ^& color 0A ^& %PYTHON_EXE% -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload"

:: Give the backend a moment to boot before the browser hits it
timeout /t 3 /nobreak >nul

:: --- 2. Start Next.js Frontend in a new window ---
echo  [2/2] Starting Next.js frontend on http://localhost:3000 ...
start "CYCLONE-X Frontend" cmd /k "title CYCLONE-X Frontend ^& color 09 ^& cd /d "%ROOT%frontend" ^& npm run dev"

:: Give Next.js ~8 seconds to compile on first run before opening the browser
echo.
echo  Waiting for services to initialize...
timeout /t 8 /nobreak >nul

:: --- 3. Open the app in the default browser ---
echo  Opening http://localhost:3000 in your browser...
start "" "http://localhost:3000"

echo.
echo  ====================================================
echo   Both servers are running in separate windows.
echo   Backend:   http://localhost:8000
echo   Frontend:  http://localhost:3000
echo   API Docs:  http://localhost:8000/docs
echo.
echo   To stop: close the two server windows, or press
echo   Ctrl+C inside each one.
echo  ====================================================
echo.
pause
