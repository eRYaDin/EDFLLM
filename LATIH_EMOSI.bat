@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto belum
.venv\Scripts\python.exe cli.py train-emotion
pause
exit /b
:belum
echo Jalankan INSTALL.bat dahulu.
pause
