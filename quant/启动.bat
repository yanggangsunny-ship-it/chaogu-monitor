@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
"C:\Users\Administrator\Desktop\Automate\buildenv314\Scripts\python.exe" app.py
if errorlevel 1 pause
