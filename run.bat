@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [설치] 가상환경을 만들고 패키지를 설치합니다. 처음 한 번만 걸립니다...
    py -3 -m venv .venv 2>nul || python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
".venv\Scripts\python.exe" -X utf8 jump_rope_counter.py %*
pause
