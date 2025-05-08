#!/usr/bin/env python
# settings.py - Application settings and configurations

import os
import torch
import tensorflow as tf

# --- GPU Settings ---
# Use CUDA if available for PyTorch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Disable GPU for TensorFlow to prevent conflicts with PyTorch
tf.config.set_visible_devices([], 'GPU')

# --- Stream Settings ---
# How many frames to buffer for each stream
BUFFER_FRAMES = 150

# --- Animal Detection Settings ---
# Minimum confidence for animal detection
DETECTION_CONFIDENCE = 0.6

# Minimum number of consecutive frames for stable detection
MIN_FRAMES_FOR_DETECTION = 3

# --- Recording Settings ---
# Default recording settings
DEFAULT_RECORDING_ENABLED = True
DEFAULT_BUFFER_SECONDS = 5
DEFAULT_RECORD_CLIP_DURATION = 60  # seconds
DEFAULT_MAX_RECORDINGS_SIZE_GB = 10
DEFAULT_MAX_RECORDINGS_FILES = 100
DEFAULT_MIN_FREE_SPACE_GB = 2

# --- Debug Settings ---
# Enable debug mode
DEBUG_MODE = True

# Directory for debug images
DEBUG_DIR = os.path.join("venv", "debug")

# --- Notification Settings ---
# Maximum number of notifications to keep in history
MAX_NOTIFICATION_HISTORY = 1000

# --- Remote Recorder Settings ---
# Check interval for remote recorders (seconds)
REMOTE_RECORDER_CHECK_INTERVAL = 30