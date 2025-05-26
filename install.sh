#!/bin/bash

set -e

python_version=$(python3 --version | cut -d " " -f 2)
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" != "3" ] || [ "$python_minor" != "10" ]; then
    echo "Error: Python 3.10 is required. Found Python $python_version"
    echo "Please install Python 3.10 before continuing."
    exit 1
fi

if ! command -v uv &> /dev/null; then
    echo "Installing UV package manager..."
    pip install uv
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS platform"
    
    if [[ $(uname -m) == 'arm64' ]]; then
        echo "Detected Apple Silicon (M1/M2/M3)"
        uv pip install -e ".[macos-silicon,dev]"
    else
        echo "Detected Intel Mac"
        uv pip install -e ".[macos,dev]"
    fi
    
    if ! python3 -c "import tkinter" &> /dev/null; then
        echo "Installing tkinter..."
        brew install python-tk@3.10
    fi
else
    echo "Detected Linux platform"
    
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA detected, installing with GPU support"
        uv pip install -e ".[linux,dev]"
    else
        echo "No CUDA detected, installing CPU-only version"
        uv pip install -e ".[dev]"
    fi
fi

mkdir -p models
if [ ! -f "models/GFPGANv1.4.pth" ]; then
    echo "Downloading GFPGANv1.4.pth model..."
    curl -L https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth -o models/GFPGANv1.4.pth
fi

if [ ! -f "models/inswapper_128_fp16.onnx" ]; then
    echo "Downloading inswapper_128_fp16.onnx model..."
    curl -L https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx -o models/inswapper_128_fp16.onnx
fi

echo "Installation complete! You can now run the application with:"
echo "  - GUI mode: python run.py"
echo "  - API mode: python app.py"
