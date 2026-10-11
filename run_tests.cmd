@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" -m unittest discover -p "test_*.py"
