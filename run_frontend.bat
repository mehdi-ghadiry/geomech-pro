@echo off
title GeoMech Pro - Frontend Dashboard
cd /d "%~dp0frontend"

echo Starting GeoMech Pro Frontend Virtual Environment...
call ..\venv\Scripts\activate.bat

echo Launching GeoMech Pro Streamlit Dashboard...
streamlit run app.py

pause
