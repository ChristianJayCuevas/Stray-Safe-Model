#!/usr/bin/env python
# stream_manager.py - Manage video streams and processing

import os
import cv2
import time
import threading
import subprocess
import numpy as np
from collections import deque, defaultdict
from datetime import datetime

# Import configuration
from config.constants import *
from config.settings import *

# Import core functions
from core.detection import detect_animals_in_frame, find_matching_tracker, create_new_tracker, process_animal_async

# Import from data
from data.database import stream_data, animal_counters

# Global dictionary for remote recorder info
remote_recorders = {}

def start_ffmpeg(stream_id):
    """
    Start FFmpeg process for HLS streaming
    
    Args:
        stream_id: ID of the stream
        
    Returns:
        tuple: (FFmpeg process, HLS directory)
    """
    hls_dir = os.path.join(os.path.dirname(__file__), '../hls_streams', stream_id)
    os.makedirs(hls_dir, exist_ok=True)
    rtmp_url = f'rtmp://127.0.0.1/live/{stream_id}'
    cmd = [
        'ffmpeg', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-r', '10', '-i', '-',
        '-c:v', 'h264_nvenc',
        '-preset', 'llhp',
        '-b:v', '1M',
        '-hls_time', '6',             
        '-hls_list_size', '10',      
        '-hls_flags', 'delete_segments+append_list+split_by_time',
        '-f', 'flv', rtmp_url
    ]

    return subprocess.Popen(cmd, stdin=subprocess.PIPE), hls_dir

def capture_frames(rtsp_url, stream_id):
    """
    Capture frames from RTSP stream
    
    Args:
        rtsp_url: RTSP URL to capture from
        stream_id: ID of the stream
    """
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Failed to open RTSP stream: {rtsp_url}")
        return

    # Check if this stream has a remote recorder
    has_remote_recorder = stream_id in REMOTE_RECORDER_IPS and USE_BUFFERED_STREAMS
    
    while True:
        # First try to get the frame from the remote recorder if available
        remote_frame = None
        if has_remote_recorder:
            remote_frame, timestamp, buffer_seconds = get_remote_frame(stream_id)
            
            # If we got a frame from the remote recorder, use it
            if remote_frame is not None:
                if stream_id not in stream_data:
                    # Stream not initialized yet, wait for process_stream to start
                    time.sleep(0.1)
                    continue
                    
                # Use the remote frame instead of capturing from RTSP
                resized = cv2.resize(remote_frame, (FRAME_WIDTH, FRAME_HEIGHT))
                stream_data[stream_id]['buffer'].append(resized)
                time.sleep(1 / 30)
                continue
        
        # If no remote frame or remote recording is disabled, use direct RTSP
        try:
            ret, frame = cap.read()
            if not ret:
                print(f"[{stream_id}] Failed to read frame")
                time.sleep(1)
                continue
    
            resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            
            if stream_id in stream_data:
                stream_data[stream_id]['buffer'].append(resized)
            
            time.sleep(1 / 30)
        except Exception as e:
            print(f"[{stream_id}] Error in capture_frames: {e}")
            time.sleep(1)

def process_stream(stream_id):
    """
    Process frames from a stream for animal detection
    
    Args:
        stream_id: ID of the stream
    """
    ffmpeg_proc, hls_dir = start_ffmpeg(stream_id)

    stream_data[stream_id].update({
        'ffmpeg': ffmpeg_proc,
        'frame_buffer': None,
        'lock': threading.Lock(),
        'hls_dir': hls_dir,
        'debug_data': {},
        'snapshots': {},
        'animal_trackers': defaultdict(lambda: []),
        'active_tracks': {},
        'tracking_history': {}
    })

    save_path = os.path.join("venv", "debug", stream_id)
    os.makedirs(save_path, exist_ok=True)

    while True:
        if len(stream_data[stream_id]['buffer']) < BUFFER_SIZE:
            time.sleep(0.1)
            continue

        frame = stream_data[stream_id]['buffer'].popleft()
        debug_snapshot = frame.copy()
        
        # Detect animals in the frame
        detections = detect_animals_in_frame(frame, stream_id)
        current_time = time.time()

        # Track all currently detected boxes
        detected_in_frame = {
            'dog': set(),
            'cat': set()
        }

        for detection in detections:
            label = detection["label"]
            confidence = detection["confidence"]
            box_center = detection["center"]
            box_area = detection["area"]
            crop = detection["crop"]
            
            # Try to match with existing tracker
            tracker, update_crop = find_matching_tracker(stream_id, label, box_center, box_area, crop, current_time)
            
            # If no matching tracker, create a new one
            if tracker is None:
                tracker = create_new_tracker(stream_id, label, box_center, box_area, crop, current_time)
                update_crop = True  # Always update crop for new trackers
            
            # Mark this tracker as detected in this frame
            detected_in_frame[label].add(tracker['id'])
            
            # Update best crop if needed
            if update_crop:
                tracker['max_area'] = box_area
                tracker['best_crop'] = crop
            
            # Save snapshot if needed
            snap_key = f"{label}{tracker['id']}"
            if not tracker['snapshot_saved'] and snap_key not in stream_data[stream_id]['snapshots']:
                snapshot_path = os.path.join(save_path, f"{stream_id}_snapshot_{snap_key}.jpg")
                cv2.imwrite(snapshot_path, debug_snapshot)
                stream_data[stream_id]['snapshots'][snap_key] = snapshot_path
                tracker['snapshot_saved'] = True

        # Check for trackers that were not detected in this frame
        for label in ['dog', 'cat']:
            if label in stream_data[stream_id]['active_tracks']:
                for track_id, tracker in list(stream_data[stream_id]['active_tracks'][label].items()):
                    if track_id not in detected_in_frame[label]:
                        time_since_last_seen = current_time - tracker['last_seen']
                        
                        # If not seen for a while, deactivate and process
                        if time_since_last_seen > TIMEOUT_SECONDS:
                            # Only process stable tracks (seen in multiple frames)
                            if tracker['frames_tracked'] >= MIN_FRAMES_FOR_DETECTION:
                                crop_key = f"{label}{tracker['id']}"
                                crop_filename = f"{stream_id}_max_{crop_key}.jpg"
                                crop_path = os.path.join(save_path, crop_filename)
                                
                                if tracker['best_crop'] is not None:
                                    # Check if we're saving a new snapshot or updating an existing one
                                    should_process = False
                                    
                                    # Save the best crop
                                    if os.path.exists(crop_path):
                                        existing = cv2.imread(crop_path)
                                        if existing is not None and tracker['best_crop'].size > existing.size:
                                            # We're updating with a larger bounding box
                                            cv2.imwrite(crop_path, tracker['best_crop'])
                                            should_process = True
                                            print(f"[{stream_id}] Updated snapshot with larger bounding box for {label}{track_id}")
                                    else:
                                        # First time saving this animal's snapshot
                                        cv2.imwrite(crop_path, tracker['best_crop'])
                                        should_process = True
                                        print(f"[{stream_id}] Created first snapshot for {label}{track_id}")
                                    
                                    # Store for debugging
                                    stream_data[stream_id]['debug_data'][f"high_conf_{crop_key}"] = tracker['best_crop']
                                    
                                    # Only run classification and matching if this is a new or updated snapshot
                                    if should_process:
                                        # Start async processing in a separate thread
                                        threading.Thread(
                                            target=process_animal_async,
                                            args=(tracker['best_crop'], stream_id, label, tracker['id']),
                                            daemon=True
                                        ).start()
                                        print(f"[{stream_id}] Started async processing for {label}{track_id}")
                                    
                                # Count animal only once when it's processed
                                animal_counters[stream_id][label] += 1
                                print(f"[{stream_id}] Processed {label}{track_id} after {tracker['frames_tracked']} frames")
                            
                            # Move to history for potential reappearance tracking
                            if label not in stream_data[stream_id]['tracking_history']:
                                stream_data[stream_id]['tracking_history'][label] = {}
                                
                            stream_data[stream_id]['tracking_history'][label][track_id] = {
                                'last_center': tracker['center'],
                                'max_area': tracker['max_area'],
                                'best_crop': tracker['best_crop'],
                                'first_seen': tracker['first_seen'],
                                'disappeared_at': current_time,
                                'snapshot_saved': tracker['snapshot_saved'],
                                'frames_tracked': tracker['frames_tracked']
                            }
                            
                            # Remove from active tracks
                            del stream_data[stream_id]['active_tracks'][label][track_id]

        # Clean up old history entries
        for label in ['dog', 'cat']:
            if label in stream_data[stream_id]['tracking_history']:
                for track_id in list(stream_data[stream_id]['tracking_history'][label].keys()):
                    history = stream_data[stream_id]['tracking_history'][label][track_id]
                    if current_time - history['disappeared_at'] > REAPPEARANCE_TIMEOUT:
                        del stream_data[stream_id]['tracking_history'][label][track_id]

        with stream_data[stream_id]['lock']:
            stream_data[stream_id]['frame_buffer'] = frame.copy()

        if ffmpeg_proc and ffmpeg_proc.stdin:
            with stream_data[stream_id]['lock']:
                _, encoded = cv2.imencode('.jpg', stream_data[stream_id]['frame_buffer'])
                ffmpeg_proc.stdin.write(encoded.tobytes())

        time.sleep(1 / 30)

def setup_streams(stream_id, video_path):
    """
    Set up static video stream for demonstration
    
    Args:
        stream_id: ID of the stream
        video_path: Path to the video file
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[{stream_id}] Failed to open static video")
        return

    ffmpeg_proc, hls_dir = start_ffmpeg(stream_id)
    stream_data[stream_id] = {
        'url': f'static://{video_path}',
        'buffer': deque(maxlen=BUFFER_SIZE),
        'ffmpeg': ffmpeg_proc,
        'frame_buffer': None,
        'lock': threading.Lock(),
        'hls_dir': hls_dir,
        'debug_data': {},
        'snapshots': {},
        'animal_trackers': defaultdict(lambda: []),
        'last_debug_crop': None,
        'active_tracks': {},  # Store currently active tracks for quick lookup
        'tracking_history': {}  # Store historical tracking data for better deduplication
    }

    save_path = os.path.join("venv", "debug", stream_id)
    os.makedirs(save_path, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        debug_snapshot = resized.copy()
        
        # Detect animals in the frame
        detections = detect_animals_in_frame(resized, stream_id)
        current_time = time.time()

        # Track all currently detected boxes
        detected_in_frame = {
            'dog': set(),
            'cat': set()
        }

        for detection in detections:
            label = detection["label"]
            confidence = detection["confidence"]
            box_center = detection["center"]
            box_area = detection["area"]
            crop = detection["crop"]
            
            # Draw detection rectangle on frame
            x1, y1, x2, y2 = detection["box"]
            cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Try to match with existing tracker
            tracker, update_crop = find_matching_tracker(stream_id, label, box_center, box_area, crop, current_time)
            
            # If no matching tracker, create a new one
            if tracker is None:
                tracker = create_new_tracker(stream_id, label, box_center, box_area, crop, current_time)
                update_crop = True  # Always update crop for new trackers
            
            # Mark this tracker as detected in this frame
            detected_in_frame[label].add(tracker['id'])
            
            # Update best crop if needed
            if update_crop:
                tracker['max_area'] = box_area
                tracker['best_crop'] = crop
            
            # Save snapshot if needed
            snap_key = f"{label}{tracker['id']}"
            if not tracker['snapshot_saved'] and snap_key not in stream_data[stream_id]['snapshots']:
                snapshot_path = os.path.join(save_path, f"{stream_id}_snapshot_{snap_key}.jpg")
                cv2.imwrite(snapshot_path, debug_snapshot)
                stream_data[stream_id]['snapshots'][snap_key] = snapshot_path
                tracker['snapshot_saved'] = True

        # Check for trackers that were not detected in this frame
        for label in ['dog', 'cat']:
            if label in stream_data[stream_id]['active_tracks']:
                for track_id, tracker in list(stream_data[stream_id]['active_tracks'][label].items()):
                    if track_id not in detected_in_frame[label]:
                        time_since_last_seen = current_time - tracker['last_seen']
                        
                        # If not seen for a while, deactivate and process
                        if time_since_last_seen > TIMEOUT_SECONDS:
                            # Only process stable tracks (seen in multiple frames)
                            if tracker['frames_tracked'] >= MIN_FRAMES_FOR_DETECTION:
                                crop_key = f"{label}{tracker['id']}"
                                crop_filename = f"{stream_id}_max_{crop_key}.jpg"
                                crop_path = os.path.join(save_path, crop_filename)
                                
                                if tracker['best_crop'] is not None:
                                    # Check if we're saving a new snapshot or updating an existing one
                                    should_process = False
                                    
                                    # Save the best crop
                                    if os.path.exists(crop_path):
                                        existing = cv2.imread(crop_path)
                                        if existing is not None and tracker['best_crop'].size > existing.size:
                                            # We're updating with a larger bounding box
                                            cv2.imwrite(crop_path, tracker['best_crop'])
                                            should_process = True
                                            print(f"[{stream_id}] Updated snapshot with larger bounding box for {label}{track_id}")
                                    else:
                                        # First time saving this animal's snapshot
                                        cv2.imwrite(crop_path, tracker['best_crop'])
                                        should_process = True
                                        print(f"[{stream_id}] Created first snapshot for {label}{track_id}")
                                    
                                    # Store for debugging
                                    stream_data[stream_id]['debug_data'][f"high_conf_{crop_key}"] = tracker['best_crop']
                                    
                                    # Only run classification and matching if this is a new or updated snapshot
                                    if should_process:
                                        # Start async processing in a separate thread
                                        threading.Thread(
                                            target=process_animal_async,
                                            args=(tracker['best_crop'], stream_id, label, tracker['id']),
                                            daemon=True
                                        ).start()
                                        print(f"[{stream_id}] Started async processing for {label}{track_id}")
                                    
                                # Count animal only once when it's processed
                                animal_counters[stream_id][label] += 1
                                print(f"[{stream_id}] Processed {label}{track_id} after {tracker['frames_tracked']} frames")
                            
                            # Move to history for potential reappearance tracking
                            if label not in stream_data[stream_id]['tracking_history']:
                                stream_data[stream_id]['tracking_history'][label] = {}
                                
                            stream_data[stream_id]['tracking_history'][label][track_id] = {
                                'last_center': tracker['center'],
                                'max_area': tracker['max_area'],
                                'best_crop': tracker['best_crop'],
                                'first_seen': tracker['first_seen'],
                                'disappeared_at': current_time,
                                'snapshot_saved': tracker['snapshot_saved'],
                                'frames_tracked': tracker['frames_tracked']
                            }
                            
                            # Remove from active tracks
                            del stream_data[stream_id]['active_tracks'][label][track_id]

        # Clean up old history entries
        for label in ['dog', 'cat']:
            if label in stream_data[stream_id]['tracking_history']:
                for track_id in list(stream_data[stream_id]['tracking_history'][label].keys()):
                    history = stream_data[stream_id]['tracking_history'][label][track_id]
                    if current_time - history['disappeared_at'] > REAPPEARANCE_TIMEOUT:
                        del stream_data[stream_id]['tracking_history'][label][track_id]

        with stream_data[stream_id]['lock']:
            stream_data[stream_id]['frame_buffer'] = resized.copy()

        if ffmpeg_proc and ffmpeg_proc.stdin:
            with stream_data[stream_id]['lock']:
                _, encoded = cv2.imencode('.jpg', stream_data[stream_id]['frame_buffer'])
                ffmpeg_proc.stdin.write(encoded.tobytes())

        time.sleep(1 / 30)

def monitor_stream(rtsp_url, stream_id):
    """
    Monitor an RTSP stream and launch processing threads when available
    
    Args:
        rtsp_url: RTSP URL to monitor
        stream_id: ID of the stream
    """
    while True:
        cap = cv2.VideoCapture(rtsp_url)
        if cap.isOpened():
            cap.release()
            print(f"[{stream_id}] Stream connected. Launching threads.")
            stream_data[stream_id] = {
                'url': rtsp_url,
                'buffer': deque(maxlen=BUFFER_SIZE)
            }
            t1 = threading.Thread(target=capture_frames, args=(rtsp_url, stream_id), daemon=True)
            t2 = threading.Thread(target=process_stream, args=(stream_id,), daemon=True)
            t1.start()
            t2.start()
            break
        else:
            print(f"[{stream_id}] Stream not available. Retrying in 30 seconds...")
            time.sleep(30)

def check_remote_recorders():
    """
    Check for remote recorders on the network and update the remote_recorders dictionary
    """
    for stream_id, client_ip in REMOTE_RECORDER_IPS.items():
        print(f"Checking for remote recorder for {stream_id} at {client_ip}...")
        remote_path = None
        
        # Platform-specific path construction
        import sys
        if sys.platform == 'win32':
            # On Windows, check the network share
            remote_path = f"\\\\{client_ip}\\{REMOTE_SHARE_NAME}\\shared"
        else:
            # On Linux, check if the directory is mounted
            remote_path = os.path.join(REMOTE_RECORDINGS_DIR, stream_id)
            
            # Create directory if it doesn't exist
            os.makedirs(remote_path, exist_ok=True)
            
            # Try to mount the remote share if not already mounted
            mount_check = subprocess.run(
                f"mount | grep {client_ip}", 
                shell=True, 
                capture_output=True, 
                text=True
            )
            
            if client_ip not in mount_check.stdout:
                try:
                    # Try to mount using CIFS/SMB
                    mount_cmd = f"mount -t cifs //{client_ip}/{REMOTE_SHARE_NAME}/shared {remote_path} -o guest"
                    subprocess.run(mount_cmd, shell=True, check=True)
                    print(f"Mounted remote recorder share from {client_ip} to {remote_path}")
                except Exception as e:
                    print(f"Failed to mount remote recorder share: {e}")
                    continue
        
        # Check if the remote directory exists and is accessible
        if remote_path and os.path.exists(remote_path):
            print(f"Found remote recorder for {stream_id} at {remote_path}")
            
            # Scan for latest frame files
            latest_frames = {}
            try:
                for item in os.listdir(remote_path):
                    if item.endswith('.jpg') and 'latest' in item:
                        # Found a latest frame file
                        recorder_id = item.replace('_latest.jpg', '')
                        
                        # Check if there's a metadata file
                        metadata_file = item.replace('.jpg', '.json')
                        metadata_path = os.path.join(remote_path, metadata_file)
                        
                        if os.path.exists(metadata_path):
                            try:
                                with open(metadata_path, 'r') as f:
                                    import json
                                    metadata = json.load(f)
                                
                                latest_frames[recorder_id] = {
                                    'frame_path': os.path.join(remote_path, item),
                                    'metadata': metadata,
                                    'last_checked': time.time()
                                }
                            except Exception as e:
                                print(f"Error reading metadata file {metadata_path}: {e}")
            except Exception as e:
                print(f"Error accessing remote path {remote_path}: {e}")
                continue
            
            if latest_frames:
                remote_recorders[stream_id] = {
                    'client_ip': client_ip,
                    'remote_path': remote_path,
                    'latest_frames': latest_frames,
                    'last_checked': time.time()
                }
                print(f"Found {len(latest_frames)} recorders for stream {stream_id}")
            else:
                print(f"No latest frames found for stream {stream_id}")

def get_remote_frame(stream_id):
    """
    Get the latest frame from a remote recorder for a stream
    
    Args:
        stream_id: ID of the stream
        
    Returns:
        tuple: (frame, timestamp, buffer_seconds) or (None, None, None) if not available
    """
    if stream_id not in remote_recorders:
        return None, None, None
    
    recorder_info = remote_recorders[stream_id]
    latest_frames = recorder_info['latest_frames']
    
    if not latest_frames:
        return None, None, None
    
    # Find the most recent frame
    latest_recorder_id = None
    latest_timestamp = 0
    
    for recorder_id, frame_info in latest_frames.items():
        metadata = frame_info.get('metadata', {})
        timestamp = metadata.get('timestamp', 0)
        
        if timestamp > latest_timestamp:
            latest_timestamp = timestamp
            latest_recorder_id = recorder_id
    
    if not latest_recorder_id:
        return None, None, None
    
    frame_info = latest_frames[latest_recorder_id]
    frame_path = frame_info['frame_path']
    metadata = frame_info.get('metadata', {})
    
    # Check if the frame file exists and is readable
    if not os.path.exists(frame_path):
        return None, None, None
    
    # Read the frame
    try:
        frame = cv2.imread(frame_path)
        timestamp = metadata.get('timestamp', time.time())
        buffer_seconds = metadata.get('buffer_seconds', 5)
        
        # Update last checked time
        frame_info['last_checked'] = time.time()
        
        return frame, timestamp, buffer_seconds
    except Exception as e:
        print(f"Error reading frame from {frame_path}: {e}")
        return None, None, None

def remote_recorder_monitor():
    """Background thread to monitor and update remote recorders"""
    while True:
        try:
            check_remote_recorders()
        except Exception as e:
            print(f"Error in remote recorder monitor: {e}")
        
        # Check every 30 seconds
        time.sleep(REMOTE_RECORDER_CHECK_INTERVAL)

# Start the remote recorder monitor in a background thread
def start_remote_recorder_monitor():
    """Start the remote recorder monitor thread"""
    threading.Thread(target=remote_recorder_monitor, daemon=True).start()