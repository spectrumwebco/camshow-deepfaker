"""
Main entry point for the Rust-powered Camshow Deepfaker application.
This allows running the package directly with `python -m camshow_deepfaker_rs`.
"""

import sys
from . import face_processing_module, video_capture_module

def main():
    """Main entry point for the application."""
    print(f"Camshow Deepfaker Rust Implementation")
    print(f"Available modules:")
    print(f"- Face Processing")
    print(f"- Video Capture")
    

if __name__ == "__main__":
    sys.exit(main())
