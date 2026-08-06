@echo off
REM Launch the app in dev mode on Windows.
cd /d "%~dp0..\..\..\typescript"

call npm run dev
