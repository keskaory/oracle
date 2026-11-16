@echo off
rem the weekly forward test run (d43). windows task scheduler starts this saturday 22:00, or as soon as
rem the laptop is on if it missed that (d30). log goes to data\forward_weekly.log.
cd /d "%~dp0"
set LOG=..\data\forward_weekly.log
set PY=..\.venv\Scripts\python.exe
rem question texts can hold non-ascii characters, and the log is a redirected file
set PYTHONIOENCODING=utf-8

rem ollama doesn't start at login on this laptop, so start it if it isn't answering
curl -s -m 5 http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    start "" /b ollama serve
    ping -n 21 127.0.0.1 >nul
)

echo === %date% %time% forecast >> %LOG%
%PY% -W ignore -u forward.py forecast --push >> %LOG% 2>&1
echo === %date% %time% score >> %LOG%
%PY% -W ignore -u forward.py score --push >> %LOG% 2>&1
echo === %date% %time% done >> %LOG%
