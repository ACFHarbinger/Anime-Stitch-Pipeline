@echo off
REM Bootstrap the dev environment on Windows.
cd /d "%~dp0..\..\.."

where uv >nul 2>nul || powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
cd python && uv sync && cd ..
cd typescript && npm install && cd ..

echo Setup complete. Activate with: python\.venv\Scripts\activate
