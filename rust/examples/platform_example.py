#!/usr/bin/env python3
"""
Platform-specific optimization example demonstrating CUDA and CoreML detection
and optimal execution provider selection.
"""

import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

try:
    from camshow_deepfaker_rs import platform_module
except ImportError:
    print("Error: Could not import Rust modules.")
    print("Make sure to build the Rust extension first with 'maturin develop' in the rust directory.")
    sys.exit(1)

def main():
    """Main function to demonstrate platform-specific optimizations."""
    print("Camshow Deepfaker Platform Detection Example")
    print("===========================================")
    
    platform_info = platform_module.PlatformInfo()
    
    print(f"Operating System: {platform_info.os_name}")
    print(f"Is Linux: {platform_info.is_linux}")
    print(f"Is macOS: {platform_info.is_macos}")
    print(f"Is Apple Silicon: {platform_info.is_apple_silicon}")
    
    print("\nHardware Acceleration:")
    print(f"CUDA Available: {platform_info.has_cuda}")
    print(f"CoreML Available: {platform_info.has_coreml}")
    
    print(f"\nRecommended Execution Provider: {platform_info.recommended_provider}")
    
    print("\nUsing individual detection functions:")
    print(f"CUDA Detection: {platform_module.py_detect_cuda()}")
    print(f"Apple Silicon Detection: {platform_module.py_detect_apple_silicon()}")
    print(f"CoreML Detection: {platform_module.py_detect_coreml()}")
    print(f"Optimal Provider: {platform_module.py_get_optimal_provider()}")
    
    print("\nPerformance Implications:")
    if platform_info.has_cuda:
        print("- CUDA detected: Expect 3-10x speedup for face processing")
        print("- Optimal for Linux environments with NVIDIA GPUs")
    elif platform_info.has_coreml:
        print("- CoreML detected: Expect 2-5x speedup for face processing")
        print("- Optimal for macOS with Apple Silicon (M1/M2/M3)")
    else:
        print("- Using CPU fallback: Performance will be limited")
        print("- Consider adding GPU support for better performance")
    
    print("\nConfiguration Recommendations:")
    if platform_info.has_cuda:
        print("- Use CUDA execution provider for optimal performance")
        print("- Ensure CUDA toolkit and cuDNN are installed")
        print("- Set environment variable: CUDA_VISIBLE_DEVICES=0")
    elif platform_info.has_coreml:
        print("- Use CoreML execution provider for optimal performance")
        print("- No additional configuration needed for Apple Silicon")
    else:
        print("- Use OpenMP for multi-threading on CPU")
        print("- Set environment variable: OMP_NUM_THREADS=<cpu_count>")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
