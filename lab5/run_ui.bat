@echo off
title Sportix AI - Sports Chatbot Server
echo ========================================================
echo   SPORTIX AI — SPORTS DOMAIN CHATBOT WEB UI
echo ========================================================
echo.
echo Launching FastAPI server on http://127.0.0.1:8000 ...
echo Press Ctrl+C in this window to stop the server.
echo.

cd /d "%~dp0"
python app.py
pause
