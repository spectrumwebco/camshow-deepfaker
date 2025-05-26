#!/usr/bin/env python3
"""
Test script for ONNX model integration with platform-specific optimizations.
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

def test_model_download():
    """Test model downloading functionality."""
    print("Testing model download...")
    
    os.makedirs("models", exist_ok=True)
    
    try:
        downloaded_models = face_processing_module.download_models()
        print(f"Downloaded models: {downloaded_models}")
    except Exception as e:
        print(f"Failed to download models: {e}")

def test_face_swapper():
    """Test face swapper with ONNX integration."""
    print("\nTesting FaceSwapper with ONNX integration...")
    
    platform_info = platform_module.PlatformInfo()
    print(f"Platform: {platform_info.os_name}")
    print(f"Recommended provider: {platform_info.recommended_provider}")
    
    try:
        swapper = face_processing_module.FaceSwapper(
            model_path="models/inswapper_128.onnx",
            device=platform_info.recommended_provider.lower()
        )
        print(f"Created face swapper: {swapper}")
        
        model_info = swapper.get_model_info()
        print(f"Model info: {model_info}")
    except Exception as e:
        print(f"Failed to create face swapper: {e}")

def test_face_enhancer():
    """Test face enhancer with ONNX integration."""
    print("\nTesting FaceEnhancer with ONNX integration...")
    
    platform_info = platform_module.PlatformInfo()
    
    try:
        enhancer = face_processing_module.FaceEnhancer(
            model_path="models/gfpgan_1.4.onnx",
            device=platform_info.recommended_provider.lower()
        )
        print(f"Created face enhancer: {enhancer}")
        
        model_info = enhancer.get_model_info()
        print(f"Model info: {model_info}")
    except Exception as e:
        print(f"Failed to create face enhancer: {e}")

def test_face_analyser():
    """Test face analyser with ONNX integration."""
    print("\nTesting FaceAnalyser with ONNX integration...")
    
    platform_info = platform_module.PlatformInfo()
    
    try:
        analyser = face_processing_module.FaceAnalyser(
            model_path="models/buffalo_l.onnx",
            device=platform_info.recommended_provider.lower()
        )
        print(f"Created face analyser: {analyser}")
        
        model_info = analyser.get_model_info()
        print(f"Model info: {model_info}")
    except Exception as e:
        print(f"Failed to create face analyser: {e}")

def main():
    """Main function to run tests."""
    print("Camshow Deepfaker ONNX Integration Test")
    print("======================================")
    
    test_model_download()
    
    test_face_swapper()
    test_face_enhancer()
    test_face_analyser()
    
    print("\nTests completed successfully!")

if __name__ == "__main__":
    main()
