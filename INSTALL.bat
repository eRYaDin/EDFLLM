@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if errorlevel 1 goto gagal
.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
if errorlevel 1 goto gagal
.venv\Scripts\python.exe cli.py train-commands
if errorlevel 1 goto gagal
echo Selesai. Jalankan JALANKAN.bat
pause
exit /b 0
:gagal
echo Instalasi gagal. Periksa Python, internet, dan pesan error di atas.
pause
exit /b 1
