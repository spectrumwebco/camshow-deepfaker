# Camshow Deepfaker Rust Implementation

This directory contains the Rust implementation of the Camshow Deepfaker application, which is being migrated from Python to Rust for significant performance improvements.

## Structure

- `src/face_processing/` - Face detection, swapping, and enhancement modules
- `src/video_capture/` - Camera and video capture functionality
- `src/ui/` - Dioxus UI components (to be implemented)

## Building

### Prerequisites

- Rust toolchain (1.70+)
- Python 3.10
- For CUDA support: CUDA toolkit 11.7+
- For Apple Silicon: Xcode 14+

### Build Commands

```bash
# Build with default features
cargo build --release

# Build with CUDA support (Linux)
cargo build --release --features cuda

# Build with CoreML support (macOS)
cargo build --release --features coreml
```

## PyO3 Integration

This implementation uses PyO3 to create Python bindings for the Rust code, allowing for a gradual migration from the existing Python codebase.

## Dioxus UI

The UI components will be implemented using Dioxus, a React-like framework for Rust that supports RSX syntax for component-based UI development.

## Performance Expectations

- Face processing: 3-10x speedup
- Video capture: 2-5x speedup
- UI rendering: 3-8x speedup (30-60fps → 120-240fps)
