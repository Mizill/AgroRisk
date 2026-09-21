@echo off
chcp 65001 >nul
rem Двойной клик по этому файлу запускает расчёт в Windows.
cd /d "%~dp0"
python run.py %*
echo.
pause
