#!/usr/bin/env python
# main.py - Main entry point for the StraySafe application

import os
import sys
import time
import threading
from flask import Flask
from flask_cors import CORS

# Import configuration
from config.constants import *
from config.settings import *

# Import API components
from api.routes import setup_routes

# Import core components
from core.stream_manager import setup_streams, monitor_stream, start_remote_recorder_monitor
from core.utils import precompute_owner_embeddings

# Import models
from models.detector import initialize_detector
from models.classifier import initialize_classifier, initialize_vgg16
from models.feature_extractor import initialize_feature_extractor
from models.breed_classifier import initialize_breed_classifier

# Import data components
from data.database import initialize_database
from data.storage import setup_storage, cleanup_hls_segments

def initialize_app():
    """Initialize the application components"""
    print("Initializing StraySafe application...")
    
    # Create necessary directories
    os.makedirs(UPLOAD_BASE_DIR, exist_ok=True)
    os.makedirs(HLS_CLEANUP_DIR, exist_ok=True)
    os.makedirs(REMOTE_RECORDINGS_DIR, exist_ok=True)
    os.chmod(HLS_CLEANUP_DIR, 0o777)
    
    # Initialize database
    print("Initializing database...")
    initialize_database()
    
    # Setup storage directories
    print("Setting up storage...")
    setup_storage()
    
    # Initialize ML models
    print("Loading machine learning models...")
    initialize_detector()
    initialize_classifier()
    initialize_vgg16()
    initialize_feature_extractor()
    initialize_breed_classifier()
    
    # Precompute embeddings for registered pets
    print("Precomputing owner embeddings...")
    precompute_owner_embeddings()
    
    # Cleanup old HLS segments
    print("Cleaning up old HLS segments...")
    deleted_count = cleanup_hls_segments()
    if deleted_count > 0:
        print(f"Deleted {deleted_count} old HLS segments")
    
    # Initialize remote recorder monitor
    print("Starting remote recorder monitor...")
    start_remote_recorder_monitor()
    
    # Initialize Flask app
    app = Flask(__name__)
    CORS(app)
    
    # Setup API routes
    print("Setting up API routes...")
    setup_routes(app)
    
    return app

def start_stream_threads():
    """Start stream monitoring and processing threads"""
    print("Starting stream threads...")
    stream_threads = []
    
    # Initialize static demo streams
    for stream_id, video_path in STATIC_VIDEO_SOURCES.items():
        print(f"Launching static stream: {stream_id} -> {video_path}")
        thread = threading.Thread(
            target=setup_streams,
            args=(stream_id, video_path), 
            daemon=True
        )
        stream_threads.append(thread)
        thread.start()
    
    # Initialize RTSP streams from IP cameras
    for ip in STREAM_IP_RANGE:
        stream_id = f'cam-{ip}'
        rtsp_url = RTSP_BASE.format(ip=ip)
        print(f"Monitoring stream: {stream_id} -> {rtsp_url}")
        thread = threading.Thread(
            target=monitor_stream, 
            args=(rtsp_url, stream_id), 
            daemon=True
        )
        stream_threads.append(thread)
        thread.start()
    
    print(f"Started {len(stream_threads)} stream threads")
    return stream_threads

def main():
    """Main entry point"""
    try:
        # Initialize application
        app = initialize_app()
        
        # Start stream processing threads
        stream_threads = start_stream_threads()
        
        # Start the Flask app
        print("Starting Flask server on port 5000...")
        app.run(host='0.0.0.0', port=5000, debug=DEBUG_MODE, use_reloader=False)
    except KeyboardInterrupt:
        print("\nStopping StraySafe application...")
        sys.exit(0)
    except Exception as e:
        print(f"Error in main application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()