@echo off
title CYCLONE-X Stop
color 0C

echo.
echo  ====================================================
echo    CYCLONE-X  ^|  Stopping all services...
echo  ====================================================
echo.

:: Kill uvicorn (backend) processes
echo  Stopping FastAPI backend (uvicorn)...
taskkill /F /FI "WINDOWTITLE eq CYCLONE-X Backend*" >nul 2>&1
taskkill /F /IM uvicorn.exe >nul 2>&1

:: Kill node (frontend) processes
echo  Stopping Next.js frontend (node)...
taskkill /F /FI "WINDOWTITLE eq CYCLONE-X Frontend*" >nul 2>&1

echo.
echo  All CYCLONE-X services stopped.
echo  ====================================================
echo.
pause
