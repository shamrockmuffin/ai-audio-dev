#!/usr/bin/env python3
"""
Script to diagnose and fix PyTorch ecosystem dependency issues
"""

import sys
import subprocess
import importlib

def check_package_version(package_name):
    """Check if a package is installed and get its version"""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name}: {version}")
        return True, version
    except ImportError as e:
        print(f"✗ {package_name}: Not installed ({e})")
        return False, None

def run_command(command):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("=== PyTorch Ecosystem Diagnostic ===\n")
    
    # Check current installations
    packages = ['torch', 'torchvision', 'torchaudio', 'transformers', 'numpy']
    
    print("Current package versions:")
    for package in packages:
        check_package_version(package)
    
    print("\n=== Checking for conflicts ===")
    
    # Try importing each package individually
    try:
        import torch
        print(f"✓ torch imports successfully (CUDA available: {torch.cuda.is_available()})")
    except Exception as e:
        print(f"✗ torch import failed: {e}")
    
    try:
        import torchvision
        print(f"✓ torchvision imports successfully")
    except Exception as e:
        print(f"✗ torchvision import failed: {e}")
    
    try:
        import torchaudio
        print(f"✓ torchaudio imports successfully")
    except Exception as e:
        print(f"✗ torchaudio import failed: {e}")
    
    try:
        from transformers import WhisperProcessor
        print(f"✓ transformers.WhisperProcessor imports successfully")
    except Exception as e:
        print(f"✗ transformers.WhisperProcessor import failed: {e}")
    
    print("\n=== Recommended fixes ===")
    print("If you see import errors above, try these commands in order:")
    print("1. .conda\\Scripts\\pip.exe uninstall torch torchvision torchaudio -y")
    print("2. .conda\\Scripts\\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("3. .conda\\Scripts\\pip.exe install --upgrade transformers")
    print("4. Restart your application")

if __name__ == "__main__":
    main() 