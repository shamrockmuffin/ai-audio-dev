@echo off
echo Starting Audio Analysis App with GPU support...
echo Using Python environment: .conda
echo.

REM Set Hugging Face token
set HUGGING_FACE_TOKEN=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

REM Set environment variables for PyTorch/Streamlit compatibility
set STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

REM Change to the project directory
cd /d "%~dp0"

REM Run the Streamlit app with optimized settings
.conda\Scripts\streamlit.exe run main.py --server.fileWatcherType=none --server.maxMessageSize=1000 --server.maxUploadSize=1000

pause 