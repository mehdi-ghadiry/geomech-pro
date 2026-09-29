@echo off
title GeoMech Pro - Backend API
cd /d "%~dp0backend"

echo Starting GeoMech Pro Backend Virtual Environment...
call ..\venv\Scripts\activate.bat

echo Launching GeoMech Pro FastAPI Backend on http://localhost:8000 ...
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
