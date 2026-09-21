@echo off
chcp 65001 >nul
rem Двойной клик открывает веб-интерфейс в браузере.
cd /d "%~dp0"
python web.py
pause
