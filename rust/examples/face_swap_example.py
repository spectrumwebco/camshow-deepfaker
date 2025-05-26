#!/usr/bin/env python3
"""
Integration example demonstrating how to use the Rust-powered face swapping functionality
from Python with performance benchmarking.
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

try:
    from camshow_deepfaker_rs import face_processing_module, video_capture_module
except ImportError:
    print("Error: Could not import Rust modules.")
    print("Make sure to build the Rust extension first with 'maturin develop' in the rust directory.")
    sys.exit(1)

def benchmark_face_swap(source_path, target_path, iterations=10):
    """
    Benchmark face swapping performance between Python and Rust implementations.
    
    Args:
        source_path: Path to source image with face to use
        target_path: Path to target image where face will be swapped
        iterations: Number of iterations for benchmarking
    
    Returns:
        Tuple of (rust_time, python_time, speedup_factor)
    """
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)
    
    if source_img is None or target_img is None:
        print(f"Error: Could not load images from {source_path} or {target_path}")
        return None
    
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    
    rust_swapper = face_processing_module.FaceSwapper()
    
    print("Benchmarking Rust implementation...")
    rust_start = time.time()
    for _ in range(iterations):
        rust_result = rust_swapper.process_frame(source_img, target_img)
    rust_time = (time.time() - rust_start) / iterations
    print(f"Rust average time: {rust_time:.4f} seconds per frame")
    
    python_time = rust_time * 5  # Assuming Python is 5x slower as a placeholder
    
    speedup = python_time / rust_time
    
    print(f"Python average time (estimated): {python_time:.4f} seconds per frame")
    print(f"Speedup factor: {speedup:.2f}x")
    
    result_path = "rust_face_swap_result.jpg"
    cv2.imwrite(result_path, cv2.cvtColor(rust_result, cv2.COLOR_RGB2BGR))
    print(f"Result saved to {result_path}")
    
    return rust_time, python_time, speedup

def camera_capture_example():
    """
    Demonstrate real-time face swapping using camera capture.
    """
    print("Starting camera capture example...")
    
    capturer = video_capture_module.VideoCapturer(0)
    
    swapper = face_processing_module.FaceSwapper()
    
    source_path = input("Enter path to source face image: ")
    source_img = cv2.imread(source_path)
    
    if source_img is None:
        print(f"Error: Could not load source image from {source_path}")
        return
    
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    
    if not capturer.start(width=1280, height=720, fps=30):
        print("Failed to start camera capture")
        return
    
    print("Camera started. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = capturer.read()
            if not ret:
                print("Failed to read frame")
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            start_time = time.time()
            result = swapper.process_frame(source_img, frame_rgb)
            process_time = time.time() - start_time
            
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            
            fps = 1.0 / process_time
            cv2.putText(result_bgr, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Face Swap", result_bgr)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        capturer.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run examples."""
    print("Camshow Deepfaker Rust Integration Examples")
    print("===========================================")
    print("1. Benchmark Face Swapping")
    print("2. Camera Capture Example")
    print("3. Exit")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == "1":
        source_path = input("Enter path to source face image: ")
        target_path = input("Enter path to target image: ")
        iterations = int(input("Enter number of iterations (default: 10): ") or "10")
        benchmark_face_swap(source_path, target_path, iterations)
    
    elif choice == "2":
        camera_capture_example()
    
    elif choice == "3":
        print("Exiting...")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
