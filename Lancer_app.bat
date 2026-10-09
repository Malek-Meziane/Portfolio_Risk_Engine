@echo off
rem Lance le Portfolio Risk Engine dans le navigateur
cd /d "%~dp0"
python -m streamlit run app.py --browser.gatherUsageStats false
pause
