@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto belum
.venv\Scripts\python.exe main.py
if errorlevel 1 pause
exit /b
:belum
echo Jalankan INSTALL.bat dahulu.
pause
