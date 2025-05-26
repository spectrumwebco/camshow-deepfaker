import os
import platform
import sys
import time
import shutil
import threading
import torch
import warnings
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import argparse
import onnxruntime
import insightface

import modules.globals
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

if 'ROCMExecutionProvider' in modules.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', help='select a source image', dest='source_path')
    parser.add_argument('-t', '--target', help='select a target image or video', dest='target_path')
    parser.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    parser.add_argument('--frame-processor', help='frame processors (choices: face_swapper, face_enhancer, ...)', dest='frame_processor', default=['face_swapper'], nargs='+')
    parser.add_argument('--keep-fps', help='keep original fps', dest='keep_fps', action='store_true', default=False)
    parser.add_argument('--keep-audio', help='keep original audio', dest='keep_audio', action='store_true', default=True)
    parser.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true', default=False)
    parser.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true', default=False)
    parser.add_argument('--map-faces', help='map source target faces', dest='map_faces', action='store_true', default=False)
    parser.add_argument('--mouth-mask', help='mask the mouth region', dest='mouth_mask', action='store_true', default=False)
    parser.add_argument('--video-encoder', help='adjust output video encoder', dest='video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9'])
    parser.add_argument('--video-quality', help='adjust output video quality', dest='video_quality', type=int, default=18, choices=range(52))
    parser.add_argument('--live-mirror', help='the live camera display as you see it in the front-facing camera frame', dest='live_mirror', action='store_true', default=False)
    parser.add_argument('--live-resizable', help='the live camera frame is resizable', dest='live_resizable', action='store_true', default=False)
    parser.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int, default=suggest_max_memory())
    parser.add_argument('--execution-provider', help='available execution provider (choices: cpu, ...)', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    parser.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    parser.add_argument('-v', '--version', action='version', version=f'{modules.metadata.name} {modules.metadata.version}')

    args = parser.parse_args()
    modules.globals.source_path = args.source_path
    modules.globals.target_path = args.target_path
    modules.globals.output_path = normalize_output_path(args.target_path, args.output_path)
    modules.globals.frame_processors = args.frame_processor
    modules.globals.keep_fps = args.keep_fps
    modules.globals.keep_audio = args.keep_audio
    modules.globals.keep_frames = args.keep_frames
    modules.globals.many_faces = args.many_faces
    modules.globals.map_faces = args.map_faces
    modules.globals.mouth_mask = args.mouth_mask
    modules.globals.video_encoder = args.video_encoder
    modules.globals.video_quality = args.video_quality
    modules.globals.live_mirror = args.live_mirror
    modules.globals.live_resizable = args.live_resizable
    modules.globals.max_memory = args.max_memory
    modules.globals.execution_providers = args.execution_provider
    modules.globals.execution_threads = args.execution_threads


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 8


def suggest_execution_providers() -> List[str]:
    execution_providers = ['cpu']
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        execution_providers.insert(0, 'cuda')
    if 'CoreMLExecutionProvider' in onnxruntime.get_available_providers():
        execution_providers.insert(0, 'coreml')
    if 'ROCMExecutionProvider' in onnxruntime.get_available_providers():
        execution_providers.insert(0, 'rocm')
    return execution_providers


def suggest_execution_threads() -> int:
    if 'ROCMExecutionProvider' in modules.globals.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    gpus = onnxruntime.get_available_providers()
    if 'CUDAExecutionProvider' in gpus or 'CoreMLExecutionProvider' in gpus or 'ROCMExecutionProvider' in gpus:
        pass
    else:
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
        except (ImportError, ModuleNotFoundError):
            pass

    if modules.globals.max_memory:
        memory = modules.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = modules.globals.max_memory * 1024 ** 6
