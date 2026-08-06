@echo off
REM Thin CLI wrapper: run the app from a terminal / shortcut on Windows.
cd /d "%~dp0..\..\.."

cd python && uv run python -m src.main %*
