#!/usr/bin/env python
# constants.py - Constant values used across the application

import os

# --- Directory Setup ---
UPLOAD_BASE_DIR = '/home/straysafe/registered_pets'
DATABASE_PATH = "/home/straysafe/registered_pets"
HLS_CLEANUP_DIR = '/var/hls'
REMOTE_RECORDINGS_DIR = '/home/straysafe/remote_recordings'
DEVICE = 'cuda'
# --- Remote Recorder Settings ---
USE_BUFFERED_STREAMS = True   # Use buffered streams from the client
REMOTE_RECORDER_IPS = {       # Map of stream IDs to client recorder IPs
    "cam-3": "10.0.0.2",      # WireGuard IP of client with recorder
    "cam-4": "10.0.0.2",
    # Add more mappings as needed
}
REMOTE_SHARE_NAME = "StraySafeRecordings"  # Network share name on Windows clients

# --- Constants ---
MATCH_THRESHOLD = 10
BUFFER_SIZE = 150
FRAME_WIDTH, FRAME_HEIGHT = 960, 544
REQUIRED_CONSECUTIVE_FRAMES = 20
DOG_CLASS_ID = 1
CAT_CLASS_ID = 0
STREAM_IP_RANGE = range(3, 11)
RTSP_BASE = 'rtsp://10.0.0.{ip}:8554/cam1'
STATIC_VIDEO_SOURCES = {
    "static-demo1": "sample_video.avi",
    "static-demo2": "sample_video2.avi",
    "static-demo3": "sample_video3.avi"
}

# --- Model File Paths ---
MODEL_DIRECTORY = 'VGG16_trained_9960.pth'
YOLO_MODEL_PATH = 'best.pt'
CNN_MODEL_PATH = "model.h5"
EFFICIENTNET_MODEL_PATH = "efficientnetb0_breed_classifier.h5"  # EfficientNetB0 breed classifier

# --- Breed Classifications ---
DOG_BREEDS = [
    "Afghan Hound", "Airedale Terrier", "Akita", "Alaskan Malamute", "American Bulldog",
    "American Pit Bull Terrier", "Australian Cattle Dog", "Australian Shepherd", "Basenji", "Basset Hound", 
    "Beagle", "Bernese Mountain Dog", "Bichon Frise", "Border Collie", "Boston Terrier", 
    "Boxer", "Bulldog", "Cavalier King Charles Spaniel", "Chihuahua", "Chow Chow", 
    "Cocker Spaniel", "Collie", "Corgi", "Dachshund", "Dalmatian", 
    "Doberman Pinscher", "English Setter", "French Bulldog", "German Shepherd", "Golden Retriever", 
    "Great Dane", "Greyhound", "Husky", "Jack Russell Terrier", "Labrador Retriever", 
    "Maltese", "Mastiff", "Miniature Schnauzer", "Newfoundland", "Pekingese", 
    "Pomeranian", "Poodle", "Pug", "Rottweiler", "Saint Bernard", 
    "Samoyed", "Shiba Inu", "Shih Tzu", "Staffordshire Bull Terrier", "Vizsla", 
    "Weimaraner", "West Highland White Terrier", "Whippet", "Yorkshire Terrier", "Mixed Breed"
]

CAT_BREEDS = [
    "Abyssinian", "American Bobtail", "American Shorthair", "Bengal", "Birman", 
    "Bombay", "British Shorthair", "Burmese", "Chartreux", "Cornish Rex", 
    "Devon Rex", "Egyptian Mau", "Exotic Shorthair", "Himalayan", "Japanese Bobtail", 
    "Maine Coon", "Manx", "Norwegian Forest Cat", "Ocicat", "Oriental", 
    "Persian", "Ragdoll", "Russian Blue", "Scottish Fold", "Siamese", 
    "Siberian", "Singapura", "Somali", "Sphynx", "Tonkinese", 
    "Turkish Angora", "Turkish Van", "Mixed Breed"
]

# --- Thresholds ---
ANIMAL_DETECTION_CONFIDENCE = 0.6
STRAY_CLASSIFICATION_THRESHOLD = 0.3
VISUAL_MATCH_THRESHOLD = 0.65
COLOR_SIMILARITY_THRESHOLD = 0.55
FEATURE_CONFIDENCE_THRESHOLD = 0.7
APPEARANCE_MATCH_THRESHOLD = 0.6
HISTORICAL_MATCH_THRESHOLD = 0.55
FEATURE_MATCH_DISTANCE_THRESHOLD = 50

# --- Tracking Parameters ---
TIMEOUT_SECONDS = 2
DIST_THRESHOLD = 80  # Distance threshold in pixels
REAPPEARANCE_TIMEOUT = 30  # Duration in seconds to consider an animal as "new" again
MIN_AREA_INCREASE = 1.2  # Minimum ratio increase in box area to update best crop

# --- Notification History Size ---
MAX_NOTIFICATION_HISTORY = 1000