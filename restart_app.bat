@echo off
echo Restarting Audio Analysis App with fixes...
echo.

REM Kill any existing Streamlit processes
taskkill /f /im streamlit.exe 2>nul
timeout /t 2 /nobreak >nul

REM Clear any cached data
if exist .cache rmdir /s /q .cache
if exist __pycache__ rmdir /s /q __pycache__

REM Set environment variables for PyTorch compatibility
set STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

echo Starting application with optimized settings...
echo.

REM Start the application
.conda\Scripts\streamlit.exe run main.py --server.fileWatcherType=none --server.maxMessageSize=1000

pause 