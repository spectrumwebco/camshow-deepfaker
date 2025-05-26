import os
import sys
import time
import tkinter
import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageOps
from typing import List, Dict, Tuple, Optional, Any, Callable
import threading
import platform

from modules import globals, metadata
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    is_image,
    is_video,
    get_temp_frame_paths,
    normalize_output_path,
    create_temp,
    resolve_relative_path,
    has_image_extension,
)
from modules.video_capture import VideoCapturer
from modules.gettext import LanguageManager

ROOT = None
POPUP = None
POPUP_LIVE = None
ROOT_HEIGHT = 700
ROOT_WIDTH = 600

def get_available_cameras():
    """Returns a list of available camera names and indices."""
    camera_indices = []
    camera_names = []

    if platform.system() == "Darwin":  # macOS specific handling
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            camera_indices.append(0)
            camera_names.append("FaceTime Camera")
            cap.release()

        for i in [1, 2]:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_indices.append(i)
                camera_names.append(f"Camera {i}")
                cap.release()
    else:
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_indices.append(i)
                camera_names.append(f"Camera {i}")
                cap.release()

    if not camera_names:
        return [], ["No cameras found"]

    return camera_indices, camera_names
