@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0ruview-wireless.ps1" %*
exit /b %errorlevel%
