#!/usr/bin/env python3
"""
Example demonstrating ONNX model integration with platform-specific optimizations.
"""

import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

try:
    from camshow_deepfaker_rs import face_processing_module, platform_module
except ImportError:
    print("Error: Could not import Rust modules.")
    print("Make sure to build the Rust extension first with 'maturin develop' in the rust directory.")
    sys.exit(1)

def test_model_manager():
    """Test the model manager functionality."""
    print("Testing ModelManager...")
    
    model_dir = "models"
    model_manager = face_processing_module.PyModelManager(model_dir)
    
    print(f"Model directory: {model_manager.model_dir}")
    
    models = ["inswapper_128.onnx", "buffalo_l.onnx", "gfpgan_1.4.onnx"]
    for model in models:
        exists = model_manager.model_exists(model)
        print(f"Model {model} exists: {exists}")
        
        if not exists:
            try:
                path = model_manager.get_model_path(model)
                print(f"Downloaded model to: {path}")
            except Exception as e:
                print(f"Failed to download model: {e}")

def test_onnx_session():
    """Test the ONNX session functionality."""
    print("\nTesting OnnxSession...")
    
    platform_info = platform_module.PlatformInfo()
    print(f"Platform: {platform_info.os_name}")
    print(f"Recommended provider: {platform_info.recommended_provider}")
    
    try:
        model_path = "models/inswapper_128.onnx"
        session = face_processing_module.PyOnnxSession(model_path, platform_info.recommended_provider.lower())
        print(f"Created session: {session}")
        print(f"Model path: {session.model_path}")
        print(f"Provider: {session.provider}")
    except Exception as e:
        print(f"Failed to create ONNX session: {e}")

def download_all_models():
    """Download all required models."""
    print("\nDownloading all models...")
    
    try:
        result = face_processing_module.download_models()
        print(f"Download result: {result}")
    except Exception as e:
        print(f"Failed to download models: {e}")

def main():
    """Main function to run examples."""
    print("Camshow Deepfaker ONNX Model Integration Example")
    print("===============================================")
    
    test_model_manager()
    
    download_all_models()
    
    test_onnx_session()
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
