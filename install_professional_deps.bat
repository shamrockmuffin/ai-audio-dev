@echo off
echo Installing professional audio processing dependencies...
echo.

REM Change to the project directory
cd /d "%~dp0"

echo Installing pyloudnorm for professional loudness measurement...
.conda\Scripts\pip.exe install pyloudnorm>=0.1.0

echo Installing python-magic for file type detection...
.conda\Scripts\pip.exe install python-magic>=0.4.27

echo Installing python-magic-bin for Windows...
.conda\Scripts\pip.exe install python-magic-bin>=0.4.14

echo Installing pytest for testing...
.conda\Scripts\pip.exe install pytest>=7.0.0 pytest-asyncio>=0.21.0

echo.
echo ✅ Professional dependencies installed successfully!
echo.
echo You can now run the application with enhanced professional features:
echo - Security validation for uploaded files
echo - Professional audio metrics (LUFS, True Peak, etc.)
echo - Intelligent caching system
echo - Streaming processing for large files
echo.
pause 