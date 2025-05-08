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
    
    while True:
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