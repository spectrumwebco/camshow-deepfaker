"""
Main entry point for the Rust-powered Camshow Deepfaker application.
This allows running the package directly with `python -m camshow_deepfaker_rs`.
"""

import sys
import importlib

def main():
    """Main entry point for the application."""
    print(f"Camshow Deepfaker Rust Implementation")
    
    try:
        from camshow_deepfaker_rs.camshow_deepfaker import face_processing_module, video_capture_module
        print(f"Available modules:")
        print(f"- Face Processing")
        print(f"- Video Capture")
        
        print(f"\nFace Processing classes:")
        for name in dir(face_processing_module):
            if not name.startswith('_'):
                print(f"  - {name}")
                
        print(f"\nVideo Capture classes:")
        for name in dir(video_capture_module):
            if not name.startswith('_'):
                print(f"  - {name}")
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("This could be because the Rust extension hasn't been built yet.")
        print("Try running 'maturin develop' in the rust directory.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
