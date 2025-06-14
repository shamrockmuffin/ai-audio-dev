@echo off
echo Installing PyAnnote.audio for Speaker Diarization...
echo.

REM Check if we're in the right directory
if not exist ".conda" (
    echo Error: .conda environment not found!
    echo Please run this script from the ai-audio project directory.
    pause
    exit /b 1
)

echo Activating conda environment...
call .conda\Scripts\activate.bat

echo.
echo Installing PyAnnote.audio dependencies...
echo This may take several minutes...
echo.

REM Install PyAnnote.audio and dependencies
.conda\Scripts\pip.exe install pyannote.audio>=3.3.0
.conda\Scripts\pip.exe install pyannote.core>=5.0.0
.conda\Scripts\pip.exe install pyannote.database>=5.1.0
.conda\Scripts\pip.exe install pyannote.metrics>=3.2.0

REM Install additional dependencies that might be needed
.conda\Scripts\pip.exe install asteroid-filterbanks
.conda\Scripts\pip.exe install speechbrain
.conda\Scripts\pip.exe install huggingface_hub

echo.
echo Verifying installation...
.conda\Scripts\python.exe -c "import pyannote.audio; print('✓ PyAnnote.audio installed successfully')"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ PyAnnote.audio installation completed successfully!
    echo.
    echo Next steps:
    echo 1. Get a HuggingFace token from: https://huggingface.co/settings/tokens
    echo 2. Accept model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1
    echo 3. Set environment variable: set HUGGING_FACE_TOKEN=your_token_here
    echo 4. Run the application: run_app.bat
) else (
    echo.
    echo ❌ Installation failed. Please check the error messages above.
)

echo.
pause 