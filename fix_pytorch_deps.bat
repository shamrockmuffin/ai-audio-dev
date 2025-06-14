@echo off
echo Fixing PyTorch ecosystem dependencies...
echo.

REM Change to the project directory
cd /d "%~dp0"

echo Step 1: Uninstalling conflicting PyTorch packages...
.conda\Scripts\pip.exe uninstall torch torchvision torchaudio -y

echo.
echo Step 2: Installing PyTorch with CUDA 12.1 support (compatible with CUDA 12.9)...
.conda\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Step 3: Upgrading transformers...
.conda\Scripts\pip.exe install --upgrade transformers

echo.
echo Step 4: Installing additional dependencies...
.conda\Scripts\pip.exe install --upgrade huggingface_hub

echo.
echo Step 5: Testing imports...
.conda\Scripts\python.exe -c "import torch; print('✓ torch:', torch.__version__)"
.conda\Scripts\python.exe -c "import torchvision; print('✓ torchvision:', torchvision.__version__)"
.conda\Scripts\python.exe -c "import torchaudio; print('✓ torchaudio:', torchaudio.__version__)"
.conda\Scripts\python.exe -c "import transformers; print('✓ transformers:', transformers.__version__)"

echo.
echo Dependencies fixed! You can now restart the application.
pause 