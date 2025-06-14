@echo off
echo Starting Audio Analysis App with GPU support...
echo Using Python environment: .conda
echo.

REM Set Hugging Face token
set HUGGING_FACE_TOKEN=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

REM Change to the project directory
cd /d "%~dp0"

REM Run the Streamlit app using the .conda environment
.conda\Scripts\streamlit.exe run main.py

pause 