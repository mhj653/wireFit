@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" wire_sample_data.py %*
