@echo off
echo Quick PyTorch Fix for Circular Import Issue
echo.

cd /d "%~dp0"

echo Testing current environment...
.conda\Scripts\python.exe test_imports.py

echo.
echo Press any key to continue with the fix, or Ctrl+C to cancel...
pause

echo.
echo Reinstalling PyTorch ecosystem...
.conda\Scripts\pip.exe uninstall torch torchvision torchaudio -y
.conda\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
.conda\Scripts\pip.exe install --upgrade transformers huggingface_hub

echo.
echo Testing after fix...
.conda\python.exe test_imports.py

echo.
echo Fix completed! Press any key to exit...
pause 