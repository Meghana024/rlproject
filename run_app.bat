@echo off
cd /d "%~dp0"
echo Starting Hospital Resource Allocation MDP Dashboard...
python -m streamlit run app.py
pause
