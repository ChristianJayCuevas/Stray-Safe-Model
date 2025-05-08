#!/usr/bin/env python
# handlers.py - API request handlers

import os
import cv2
import json
import time
import base64
import numpy as np
from flask import jsonify, send_from_directory
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import shutil

# Import from other modules
from config.constants import *
from config.settings import *
from core.detection import classify_and_match, save_debug_images
from core.notification import notify_owner, notify_pound
from models.classifier import transform, vgg_model
from data.database import owner_embeddings, detected_animals_log, notification_history, stream_data, animal_counters
from core.utils import match_snapshot_to_owner, precompute_owner_embeddings, remove_green_border, log_animal_detection
from core.stream_manager import check_remote_recorders, get_remote_frame, remote_recorders

def handle_detected_animals(request):
    """Handle request for detected animals with filtering options"""
    # Get query parameters
    stream_id = request.args.get('stream_id')
    animal_type = request.args.get('animal_type')  # 'dog' or 'cat'
    classification = request.args.get('classification')  # 'stray' or 'not_stray'
    notification_type = request.args.get('notification_type')  # 'owner_notification' or 'pound_notification'
    notification_case = request.args.get('notification_case')  # 'stray_registered', 'stray_unregistered', etc.
    owner_id = request.args.get('owner_id')  # Filter by owner_id
    limit = int(request.args.get('limit', 100))  # Default to 100 results
    
    # If no detections exist for the requested stream_id, run save_debug_images to populate the log
    if stream_id and not any(d['stream_id'] == stream_id for d in detected_animals_log):
        result = save_debug_images(stream_id)
        # If debug analysis was successful, wait a moment for the log to update
        if result:
            time.sleep(0.2)  # Short delay to ensure the log is updated
    # If no specific stream_id was provided, check all available streams
    elif not stream_id:
        # Get list of all stream IDs
        all_streams = list(stream_data.keys())
        for sid in all_streams:
            # Only analyze streams with no detections in the log
            if not any(d['stream_id'] == sid for d in detected_animals_log):
                save_debug_images(sid)
    
    # Filter results based on query parameters
    filtered_results = detected_animals_log.copy()
    
    if stream_id:
        filtered_results = [d for d in filtered_results if d['stream_id'] == stream_id]
    
    if animal_type:
        filtered_results = [d for d in filtered_results if d['animal_type'] == animal_type]
    
    if classification:
        filtered_results = [d for d in filtered_results if d['classification'] == classification]
        
    if notification_type:
        filtered_results = [d for d in filtered_results if d.get('notification_type') == notification_type]
        
    if notification_case:
        filtered_results = [d for d in filtered_results if d.get('notification_case') == notification_case]
    
    # New filter by owner_id
    if owner_id:
        filtered_results = [d for d in filtered_results if d.get('owner_id') == owner_id]
    
    # Limit the number of results
    filtered_results = filtered_results[:limit]
    
    # Add image URLs for frontend display
    for result in filtered_results:
        # Add image URL
        if result.get('image_path'):
            result['image_url'] = f"/api2/detected-img/{result['stream_id']}/{os.path.basename(result['image_path'])}"
        
        # Ensure all fields are present (for backward compatibility with older entries)
        if 'notification_case' not in result:
            # Determine notification case based on classification and match status
            is_stray = result.get('classification') == 'stray'
            has_match = bool(result.get('match'))
            
            # Set notification case and type if not already present
            if is_stray and has_match:
                result['notification_case'] = 'stray_registered'
                result['notification_type'] = 'owner_notification'
            elif is_stray and not has_match:
                result['notification_case'] = 'stray_unregistered'
                result['notification_type'] = 'pound_notification'
            elif not is_stray and has_match:
                result['notification_case'] = 'not_stray_registered'
                result['notification_type'] = 'owner_notification'
            elif not is_stray and not has_match:
                result['notification_case'] = 'not_stray_unregistered'
                result['notification_type'] = 'pound_notification'
        
    return jsonify({
        "count": len(filtered_results),
        "detected_animals": filtered_results
    })

def handle_predict_animal(request):
    """Predict if an animal is a stray"""
    data = request.json
    image_path = data.get("image_path")
    animal_type = data.get("animal_type")
    
    if not os.path.exists(image_path):
        return jsonify({"error": "Image not found"}), 400
        
    image = cv2.imread(image_path)
    cropped = remove_green_border(image)

    # Import CNN model from models
    from models.classifier import cnn_model, preprocess_image
    
    prediction = cnn_model.predict(preprocess_image(cropped))
    is_stray = prediction[0] >= 0.3

    if not is_stray:
        return jsonify({"predicted_label": "not stray", "action": "none"})

    best_match = match_snapshot_to_owner(cropped)
    if best_match and best_match.get('owner_id'):
        notify_owner(best_match.get('owner_id'), None, {"image_path": image_path})
        return jsonify({
            "predicted_label": "stray", 
            "match_found": True, 
            "animal_id": best_match.get('match'),
            "owner_id": best_match.get('owner_id')
        })
    else:
        notify_pound(image_path)
        return jsonify({"predicted_label": "stray", "match_found": False})

def handle_all_matches(stream_id):
    """Get all potential matches for a detected animal"""
    debug_dir = os.path.join("venv", "debug", stream_id)
    abs_debug_dir = os.path.abspath(debug_dir)
    os.makedirs(abs_debug_dir, exist_ok=True)

    if not os.path.exists(abs_debug_dir):
        return jsonify({"error": "Debug directory not found"}), 404

    # Find the latest max snapshot file matching pattern stream_id_max_dog1.jpg or stream_id_max_cat1.jpg
    max_files = [f for f in os.listdir(abs_debug_dir) if f.startswith(f"{stream_id}_max_") and 
                (f.find("dog") != -1 or f.find("cat") != -1) and f.endswith(".jpg")]
    
    if not max_files:
        return jsonify({"error": "No animal snapshots found"}), 404

    # Get the latest file by modification time
    latest_file = max(max_files, key=lambda f: os.path.getmtime(os.path.join(abs_debug_dir, f)))
    high_conf_path = os.path.join(abs_debug_dir, latest_file)
    high_conf_frame = cv2.imread(high_conf_path)

    if high_conf_frame is None:
        return jsonify({"error": "Could not read image file"}), 500
    
    # Determine animal type from filename
    animal_type = "dog" if "dog" in latest_file else "cat"
    animal_id = latest_file.split("_")[-1].split(".")[0]  # Extract the ID (e.g., "dog1" -> "1")
    
    # Clean the image
    cleaned = remove_green_border(high_conf_frame)
    
    # Find all potential matches
    match_result = match_snapshot_to_owner(cleaned)
    if not match_result or 'all_matches' not in match_result or not match_result['all_matches']:
        return jsonify({"error": "No matches found", "animal_type": animal_type, "animal_id": animal_id}), 404
    
    # Directory to save comparison images
    comparisons_dir = os.path.join(abs_debug_dir, "comparisons")
    os.makedirs(comparisons_dir, exist_ok=True)
    
    # Create side-by-side comparisons for all matches
    comparison_results = []
    
    for idx, match_info in enumerate(match_result['all_matches']):
        owner_img_path = match_info['path']
        owner_img = cv2.imread(owner_img_path)
        
        if owner_img is None:
            continue
            
        # Create side-by-side comparison
        comparison_path = os.path.join(comparisons_dir, f"comparison_{idx+1}_{os.path.basename(owner_img_path)}")
        
        # Resize images to same height
        h1, w1 = cleaned.shape[:2]
        h2, w2 = owner_img.shape[:2]
        target_height = max(h1, h2)
        
        resized1 = cv2.resize(cleaned, (int(w1 * target_height / h1), target_height)) if h1 != target_height else cleaned
        resized2 = cv2.resize(owner_img, (int(w2 * target_height / h2), target_height)) if h2 != target_height else owner_img
        
        # Add match information to the comparison image
        color_score = match_info.get('color_score', 0)
        combined_score = match_info.get('combined_score', 0)
        info_text = f"Match: {match_info['filename']} (Color: {color_score:.2f}, Combined: {combined_score:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Create a blank info bar
        info_bar = np.ones((40, resized1.shape[1] + resized2.shape[1], 3), dtype=np.uint8) * 255
        cv2.putText(info_bar, info_text, (10, 30), font, 0.7, (0, 0, 0), 2)
        
        # Stack the images vertically with the info bar
        stacked = np.vstack([info_bar, np.hstack((resized1, resized2))])
        cv2.imwrite(comparison_path, stacked)
        
        comparison_results.append({
            "rank": idx + 1,
            "filename": match_info['filename'],
            "combined_score": match_info['combined_score'],
            "color_score": match_info.get('color_score', 0),
            "feature_score": match_info.get('feature_score', 0),
            "method": match_info.get('method', 'visual'),
            "comparison_url": f"/api2/debug-img/{stream_id}/comparisons/{os.path.basename(comparison_path)}"
        })
    
    return jsonify({
        "animal_type": animal_type,
        "animal_id": animal_id,
        "original_file": latest_file,
        "matches_count": len(comparison_results),
        "matches": comparison_results
    })

def handle_notification_list(request):
    """Get a list of all notifications with filtering options"""
    
    # Parse query parameters
    notification_type = request.args.get('type')  # 'owner_notification' or 'pound_notification'
    limit = int(request.args.get('limit', 50))
    stream_id = request.args.get('stream_id')
    animal_type = request.args.get('animal_type')
    notification_case = request.args.get('case')  # stray_registered, not_stray_unregistered, etc.
    
    # Filter notifications based on parameters
    filtered_notifications = notification_history.copy()
    
    if notification_type:
        filtered_notifications = [n for n in filtered_notifications if n.get('type') == notification_type]
    
    if notification_case:
        filtered_notifications = [n for n in filtered_notifications if 
                                n.get('animal_info', {}).get('notification_case') == notification_case]
    
    if stream_id:
        # Check both the direct stream_id and the one in animal_info
        filtered_notifications = [n for n in filtered_notifications if 
                               (n.get('stream_id') == stream_id or 
                                n.get('animal_info', {}).get('stream_id') == stream_id)]
    
    if animal_type:
        filtered_notifications = [n for n in filtered_notifications if 
                                n.get('animal_info', {}).get('animal_type') == animal_type]
    
    # Limit results
    limited_notifications = filtered_notifications[:limit]
    
    # Add display-friendly stream_id to each notification if not already present
    for notification in limited_notifications:
        if 'stream_id' not in notification:
            notification['stream_id'] = notification.get('animal_info', {}).get('stream_id', 'unknown')
    
    return jsonify({
        "count": len(limited_notifications),
        "notifications": limited_notifications
    })

def handle_notification_stats():
    """Get statistics about notifications"""
    
    # Count by notification type
    by_type = {
        "owner_notification": 0,
        "pound_notification": 0
    }
    
    # Count by notification case
    by_case = {
        "stray_registered": 0,
        "stray_unregistered": 0,
        "not_stray_registered": 0,
        "not_stray_unregistered": 0
    }
    
    # Count by stream
    by_stream = {}
    
    # Count by animal type
    by_animal_type = {
        "dog": 0,
        "cat": 0
    }
    
    for notification in notification_history:
        # Count by type
        ntype = notification.get('type')
        if ntype in by_type:
            by_type[ntype] += 1
        
        # Get animal info
        animal_info = notification.get('animal_info', {})
        
        # Get stream_id either from the notification directly or from animal_info
        stream_id = notification.get('stream_id') or animal_info.get('stream_id', 'unknown')
        
        # Count by case
        case = animal_info.get('notification_case')
        if case in by_case:
            by_case[case] += 1
        
        # Count by stream
        if stream_id:
            if stream_id not in by_stream:
                by_stream[stream_id] = 0
            by_stream[stream_id] += 1
        
        # Count by animal type
        atype = animal_info.get('animal_type')
        if atype in by_animal_type:
            by_animal_type[atype] += 1
    
    return jsonify({
        "total": len(notification_history),
        "by_type": by_type,
        "by_case": by_case,
        "by_stream": by_stream,
        "by_animal_type": by_animal_type,
        "recent_streams": list(by_stream.keys())[:5]  # Include 5 most recent streams for quick reference
    })

def handle_notification_details(notification_id):
    """Get detailed information about a specific notification"""
    
    for notification in notification_history:
        if notification.get('id') == notification_id:
            return jsonify(notification)
    
    return jsonify({"error": "Notification not found"}), 404

def handle_database_status():
    """Debug endpoint to check database status"""
    # Re-load database to ensure fresh data
    count = precompute_owner_embeddings()
    
    return jsonify({
        "database_path": DATABASE_PATH,
        "files_loaded": count,
        "sample_files": list(owner_embeddings.keys())[:5] if owner_embeddings else [],
        "status": "ok" if count > 0 else "error"
    })

def handle_test_notification(request):
    """Test endpoint to create a notification for debugging purposes"""
    stream_id = request.args.get('stream_id', 'test-stream')
    animal_type = request.args.get('animal_type', 'dog')
    notify_type = request.args.get('notify_type', 'owner')  # 'owner' or 'pound'
    
    # Create dummy animal info
    animal_info = {
        "stream_id": stream_id,
        "animal_type": animal_type,
        "animal_id": "test1",
        "classification": "not_stray" if notify_type == "owner" else "stray",
        "prediction_score": 0.2 if notify_type == "owner" else 0.8,
        "is_stray": notify_type != "owner",
        "match": "test_owner.jpg" if notify_type == "owner" else None,
        "match_score": 0.75 if notify_type == "owner" else 0,
        "match_method": "test" if notify_type == "owner" else "none",
        "timestamp": datetime.now().isoformat(),
        "detection_id": f"{stream_id}_{animal_type}test1_{int(time.time())}",
        "notification_case": "not_stray_registered" if notify_type == "owner" else "stray_unregistered"
    }
    
    notification = None
    
    # Create a notification
    if notify_type == "owner":
        # Create owner notification
        notification = notify_owner("test_owner.jpg", None, animal_info)
    else:
        # Create pound notification
        tmp_path = os.path.join("venv", "tmp", f"test_{animal_type}_{stream_id}_{int(time.time())}.jpg")
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        
        # Create a blank image if needed for testing
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.imwrite(tmp_path, test_img)
        
        notification = notify_pound(tmp_path, animal_info)
    
    return jsonify({
        "status": "success",
        "message": f"Created test {notify_type} notification for {stream_id}",
        "notification": notification
    })

def handle_upload_pet_image(request):
    """Handle upload of a pet image"""
    file = request.files.get("file")
    username = request.form.get("username")

    if not file or not username:
        return jsonify({"status": "error", "message": "Missing file or username"}), 400

    try:
        # Open the uploaded image using PIL
        image = Image.open(file.stream).convert("RGB")
        
        # Save a temporary copy of the image for OpenCV processing
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"temp_upload_{int(time.time())}.jpg")
        image.save(temp_path)
        
        # Load with OpenCV for breed classification
        cv_image = cv2.imread(temp_path)
        
        # Classify as dog or cat using VGG16
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = vgg_model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probs, 1)
        
        animal_type = "cat" if prediction.item() == 0 else "dog"

        if confidence.item() < 0.7:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return jsonify({
                "status": "rejected",
                "message": f"Low confidence ({confidence.item():.2f}) — Please upload a clearer image."
            })

        # Classify the breed using EfficientNetB0
        from models.breed_classifier import predict_breed
        breed_name, breed_confidence = predict_breed(cv_image, animal_type)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Create user directory if it doesn't exist
        user_dir = os.path.join(UPLOAD_BASE_DIR, username)
        
        # If this is a new directory, copy contents from with_leash
        if not os.path.exists(user_dir):
            os.makedirs(user_dir, exist_ok=True)
            with_leash_dir = os.path.join(UPLOAD_BASE_DIR, "with_leash")
            if os.path.exists(with_leash_dir):
                for item in os.listdir(with_leash_dir):
                    src = os.path.join(with_leash_dir, item)
                    dst = os.path.join(user_dir, item)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                    elif os.path.isdir(src):
                        shutil.copytree(src, dst)
        else:
            os.makedirs(user_dir, exist_ok=True)

        # Save the uploaded image
        filename = secure_filename(file.filename)
        save_path = os.path.join(user_dir, filename)
        image.save(save_path)

        # Also update the embeddings for this image for future matching
        from core.detection import get_image_embedding
        
        # Wait a short moment to ensure file is saved
        time.sleep(0.1)
        
        try:
            # Load the saved image and compute embedding
            saved_img = cv2.imread(save_path)
            if saved_img is not None:
                embedding = get_image_embedding(saved_img)
                
                # Save the embedding with owner information
                owner_embeddings[filename] = {
                    "embedding": embedding,
                    "owner_id": username,
                    "breed": breed_name,
                    "animal_type": animal_type
                }
                print(f"Added embedding for {filename} - owner: {username}, breed: {breed_name}")
        except Exception as e:
            print(f"Error computing embedding: {e}")

        return jsonify({
            "status": "success",
            "animal_type": animal_type,
            "type_confidence": round(confidence.item(), 4),
            "breed": breed_name,
            "breed_confidence": round(float(breed_confidence), 4),
            "file_url": f"/api2/pet-image/{username}/{filename}"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def handle_get_pet_image(username, filename):
    """Serve a pet image"""
    user_dir = os.path.join(UPLOAD_BASE_DIR, secure_filename(username))
    return send_from_directory(user_dir, filename)

def handle_remote_recorders_status(request):
    """Get status of remote recorders"""
    
    # Refresh remote recorders if requested
    refresh = request.args.get('refresh', '').lower() == 'true'
    if refresh:
        try:
            check_remote_recorders()
        except Exception as e:
            print(f"Error refreshing remote recorders: {e}")
    
    # Convert timestamps to human-readable format
    recorder_status = {}
    for stream_id, info in remote_recorders.items():
        recorder_status[stream_id] = {
            'client_ip': info['client_ip'],
            'remote_path': info['remote_path'],
            'last_checked': datetime.fromtimestamp(info['last_checked']).isoformat(),
            'recorder_count': len(info['latest_frames']),
            'recorders': {}
        }
        
        # Add information about each recorder
        for recorder_id, frame_info in info['latest_frames'].items():
            metadata = frame_info.get('metadata', {})
            timestamp = metadata.get('timestamp', 0)
            buffer_seconds = metadata.get('buffer_seconds', 0)
            
            recorder_status[stream_id]['recorders'][recorder_id] = {
                'timestamp': datetime.fromtimestamp(timestamp).isoformat() if timestamp else None,
                'buffer_seconds': buffer_seconds,
                'last_checked': datetime.fromtimestamp(frame_info['last_checked']).isoformat(),
                'frame_available': os.path.exists(frame_info['frame_path']),
                'stream_url': metadata.get('stream_url', 'unknown'),
                'frame_dimensions': f"{metadata.get('frame_width', '?')}x{metadata.get('frame_height', '?')}"
            }
    
    return jsonify({
        'remote_recorders_enabled': USE_BUFFERED_STREAMS,
        'recorder_count': len(remote_recorders),
        'recorders': recorder_status
    })

def handle_toggle_remote_recorders(request):
    """Enable or disable remote recorders"""
    global USE_BUFFERED_STREAMS
    
    data = request.json or {}
    enabled = data.get('enabled')
    
    if enabled is None:
        return jsonify({"error": "Missing 'enabled' parameter"}), 400
    
    # Toggle the flag
    USE_BUFFERED_STREAMS = bool(enabled)
    
    return jsonify({
        "remote_recorders_enabled": USE_BUFFERED_STREAMS,
        "message": f"Remote recorders {'enabled' if USE_BUFFERED_STREAMS else 'disabled'}"
    })

def handle_remote_recorder_snapshot(stream_id):
    """Get a snapshot from a remote recorder"""
    
    # Check if the stream exists and has a remote recorder
    if stream_id not in REMOTE_RECORDER_IPS:
        return jsonify({"error": f"No remote recorder configured for stream {stream_id}"}), 404
    
    # Try to get a frame from the remote recorder
    try:
        frame, timestamp, buffer_seconds = get_remote_frame(stream_id)
        
        if frame is None:
            return jsonify({"error": "Failed to get snapshot from remote recorder"}), 500
            
        # Convert to JPEG and base64 encode
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            return jsonify({"error": "Failed to encode image"}), 500
            
        encoded = base64.b64encode(jpeg.tobytes()).decode('utf-8')
        
        return jsonify({
            "stream_id": stream_id,
            "timestamp": timestamp,
            "buffer_seconds": buffer_seconds,
            "image": encoded
        })
    except Exception as e:
        return jsonify({"error": f"Error getting snapshot: {str(e)}"}), 500

def handle_detected_image(stream_id, filename):
    """Serve detected animal images from debug directory"""
    debug_dir = os.path.join("venv", "debug", stream_id)
    return send_from_directory(os.path.abspath(debug_dir), filename)

def handle_animal_stats():
    """Get statistics about detected animals"""
    stats = {
        "total_detections": len(detected_animals_log),
        "animal_types": {},
        "classifications": {
            "stray": 0,
            "not_stray": 0
        },
        "streams": {},
        "notification_cases": {
            "stray_registered": 0,
            "stray_unregistered": 0,
            "not_stray_registered": 0,
            "not_stray_unregistered": 0
        }
    }
    
    # Count by animal type
    for animal in detected_animals_log:
        animal_type = animal.get('animal_type', 'unknown')
        if animal_type not in stats["animal_types"]:
            stats["animal_types"][animal_type] = 0
        stats["animal_types"][animal_type] += 1
        
        # Count by classification
        classification = animal.get('classification', 'unknown')
        if classification in stats["classifications"]:
            stats["classifications"][classification] += 1
            
        # Count by stream
        stream_id = animal.get('stream_id', 'unknown')
        if stream_id not in stats["streams"]:
            stats["streams"][stream_id] = 0
        stats["streams"][stream_id] += 1
        
        # Count by notification case
        notification_case = animal.get('notification_case')
        if notification_case in stats["notification_cases"]:
            stats["notification_cases"][notification_case] += 1
    
    return jsonify(stats)

def handle_debug_pipeline(stream_id):
    """Debug detection pipeline for a stream"""
    if stream_id not in stream_data:
        return jsonify({"error": f"Stream {stream_id} not found"}), 404
    
    # Run detection pipeline on this stream
    result = save_debug_images(stream_id)
    
    if not result:
        return jsonify({"error": "Failed to process debug pipeline"}), 500
    
    # Get all debug images for this stream
    debug_dir = os.path.join("venv", "debug", stream_id)
    abs_debug_dir = os.path.abspath(debug_dir)
    
    if not os.path.exists(abs_debug_dir):
        return jsonify({"error": "Debug directory not found"}), 404
    
    # List all debug images
    debug_files = [f for f in os.listdir(abs_debug_dir) if f.endswith('.jpg')]
    
    # Generate URLs for frontend
    debug_urls = {
        f: f"/api2/debug-img/{stream_id}/{f}" for f in debug_files
    }
    
    # Return debug info
    return jsonify({
        "stream_id": stream_id,
        "debug_images": debug_urls,
        "pipeline_data": stream_data[stream_id].get('debug_data', {})
    })

def handle_debug_image(stream_id, filename):
    """Serve debug images from debug directory"""
    debug_dir = os.path.join("venv", "debug", stream_id)
    return send_from_directory(os.path.abspath(debug_dir), filename)

def handle_streams_list():
    """Get information about active streams"""
    streams_info = {}
    
    for stream_id, data in stream_data.items():
        # Get information about this stream
        stream_info = {
            "id": stream_id,
            "status": "active" if data.get('buffer') and len(data.get('buffer', [])) > 0 else "inactive",
            "buffer_size": len(data.get('buffer', [])),
            "has_remote_recorder": stream_id in REMOTE_RECORDER_IPS,
            "remote_recorder_status": "unknown"
        }
        
        # Check if this stream has a remote recorder
        if stream_id in remote_recorders:
            stream_info["remote_recorder_status"] = remote_recorders[stream_id].get("status", "unknown")
            stream_info["remote_recorder_last_seen"] = remote_recorders[stream_id].get("last_seen", 0)
        
        streams_info[stream_id] = stream_info
    
    return jsonify(streams_info)

def handle_animal_counters():
    """Get animal counters for each stream"""
    return jsonify(animal_counters)

def handle_video_snapshot(stream_id):
    """Get a snapshot of the current video frame"""
    if stream_id not in stream_data:
        return jsonify({"error": f"Stream {stream_id} not found"}), 404
    
    # Try to get a frame from the buffer
    buffer = stream_data[stream_id].get('buffer', None)
    if not buffer or len(buffer) == 0:
        return jsonify({"error": "No frames in buffer"}), 404
    
    # Get the most recent frame
    frame = buffer[-1].copy()
    
    # Convert to JPEG
    ret, jpeg = cv2.imencode('.jpg', frame)
    
    if not ret:
        return jsonify({"error": "Failed to encode image"}), 500
    
    # Return as base64 encoded image
    encoded = base64.b64encode(jpeg.tobytes()).decode('utf-8')
    
    return jsonify({
        "stream_id": stream_id,
        "timestamp": time.time(),
        "image": encoded
    })

def handle_hls_file(stream_id, filename):
    """Serve HLS stream files"""
    hls_dir = os.path.join(HLS_CLEANUP_DIR, stream_id)
    
    if not os.path.exists(hls_dir):
        return jsonify({"error": f"HLS directory not found for stream {stream_id}"}), 404
    
    return send_from_directory(hls_dir, filename)

def handle_breed_classification(request):
    """Handle breed classification for an uploaded image or image path"""
    try:
        data = request.json
        image_path = data.get("image_path")
        animal_type = data.get("animal_type")  # 'dog' or 'cat', optional
        image_data = data.get("image_data")  # Base64 encoded image, optional
        
        # Check if we have either an image path or image data
        if not image_path and not image_data:
            return jsonify({"error": "Either image_path or image_data is required"}), 400
            
        # Load the image
        if image_path:
            if not os.path.exists(image_path):
                return jsonify({"error": "Image file not found"}), 404
                
            image = cv2.imread(image_path)
        else:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
        if image is None:
            return jsonify({"error": "Failed to load image"}), 400
            
        # Clean the image (remove any green borders)
        cleaned_image = remove_green_border(image)
        
        # If animal type is not specified, determine it
        if not animal_type:
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray(cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2RGB))
            from models.classifier import classify_animal_type
            animal_type, type_confidence = classify_animal_type(pil_image)
        else:
            type_confidence = 1.0
            
        # Predict breed
        from models.breed_classifier import predict_breed
        breed_name, breed_confidence = predict_breed(cleaned_image, animal_type)
        
        # Return the results
        return jsonify({
            "animal_type": animal_type,
            "type_confidence": float(type_confidence) if 'type_confidence' in locals() else 1.0,
            "breed": breed_name,
            "breed_confidence": float(breed_confidence)
        })
            
    except Exception as e:
        return jsonify({"error": f"Error classifying breed: {str(e)}"}), 500