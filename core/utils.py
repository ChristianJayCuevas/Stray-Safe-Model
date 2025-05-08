#!/usr/bin/env python
# utils.py - Utility functions for the StraySafe application

import os
import cv2
import time
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# Import configuration
from config.constants import *
from data.database import owner_embeddings, detected_animals_log

def remove_green_border(image):
    """
    Filters out green background used in bounding boxes
    
    Args:
        image: OpenCV image with potential green border
        
    Returns:
        Filtered image with green border removed
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([55, 80, 70])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return image[y:y+h, x:x+w]
    return image

def get_image_embedding(img):
    """
    Get embedding vector for an image using the feature extractor
    
    Args:
        img: OpenCV image
        
    Returns:
        Feature embedding vector
    """
    # Import feature extractor to avoid circular imports
    from models.feature_extractor import feature_extractor, preprocess_input
    
    # Convert BGR (OpenCV) to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (224, 224))
    x = np.expand_dims(resized, axis=0).astype('float32')
    x = preprocess_input(x)
    return feature_extractor.predict(x)[0]

def precompute_owner_embeddings():
    """
    Load pet images from user-specific folders under DATABASE_PATH and 
    precompute their embeddings with associated owner IDs.
    
    Returns:
        int: Number of images processed
    """
    print(f"Loading pet images from: {DATABASE_PATH}")
    count = 0

    try:
        if not os.path.exists(DATABASE_PATH):
            print(f"WARNING: Directory {DATABASE_PATH} does not exist!")
            os.makedirs(DATABASE_PATH, exist_ok=True)
            print(f"Created directory {DATABASE_PATH}")
            return 0

        # Iterate over each user folder
        for owner_id in os.listdir(DATABASE_PATH):
            user_folder_path = os.path.join(DATABASE_PATH, owner_id)
            if not os.path.isdir(user_folder_path):
                continue

            for fname in os.listdir(user_folder_path):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                full_path = os.path.join(user_folder_path, fname)
                img = cv2.imread(full_path)

                if img is None:
                    print(f"[WARN] Failed to read image: {full_path}")
                    continue

                embedding = get_image_embedding(img)

                # Save with embedding + owner_id
                owner_embeddings[fname] = {
                    "embedding": embedding,
                    "owner_id": owner_id
                }

                count += 1

        print(f"Successfully loaded {count} pet images and computed embeddings.")
    except Exception as e:
        print(f"[ERROR] Failed to load pet database: {str(e)}")

    return count

def log_animal_detection(stream_id, animal_type, animal_id, classification_result, match_result=None, image_path=None):
    """
    Log a detected animal with detailed information for tracking purposes
    
    Args:
        stream_id: ID of the stream
        animal_type: Type of animal ('dog' or 'cat')
        animal_id: ID of the animal
        classification_result: Result of classification ('stray' or 'not_stray')
        match_result: Result of matching against database
        image_path: Path to the image file
        
    Returns:
        dict: Created detection entry
    """
    timestamp = datetime.now().isoformat()
    
    # Get owner_id from match_result if available
    owner_id = match_result.get("owner_id") if match_result else None
    
    detection_entry = {
        "id": f"{stream_id}_{animal_type}{animal_id}_{int(time.time())}",
        "stream_id": stream_id,
        "animal_type": animal_type,
        "animal_id": animal_id,
        "timestamp": timestamp,
        "classification": classification_result,
        "image_path": image_path,
        "owner_id": owner_id
    }
    
    # Add match information if available
    if match_result:
        detection_entry["match"] = match_result.get("match")
        detection_entry["match_score"] = match_result.get("score", 0)
        detection_entry["match_method"] = match_result.get("method", "none")
    
    # Add to the beginning of the list for most recent first
    detected_animals_log.insert(0, detection_entry)
    
    # Keep the log at a reasonable size
    if len(detected_animals_log) > MAX_NOTIFICATION_HISTORY:
        detected_animals_log.pop()
    
    return detection_entry

def match_snapshot_to_owner(snapshot_img, threshold=0.65):
    """
    Match a detected animal to owner records using visual similarities.
    Uses both color histogram + ORB feature detection, falls back to CNN embeddings.
    
    Args:
        snapshot_img: Image of the detected animal
        threshold: Similarity threshold for matching
        
    Returns:
        dict: Match results including owner_id if a match is found
    """
    best_match = None
    best_score = -1
    match_method = "unknown"
    match_owner_id = None

    all_matches = []
    visual_matches = []
    embedding_matches = []

    # Setup ORB feature detector
    orb = cv2.ORB_create(nfeatures=10000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for fname, data in owner_embeddings.items():
        owner_id = data.get("owner_id")
        db_embedding = data.get("embedding")
        path = os.path.join(DATABASE_PATH, owner_id, fname)
        owner_img = cv2.imread(path)

        if owner_img is None:
            continue

        try:
            # Color histogram comparison
            hsv1 = cv2.cvtColor(snapshot_img, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(owner_img, cv2.COLOR_BGR2HSV)

            h_bins, s_bins = 50, 60
            histSize = [h_bins, s_bins]
            ranges = [0, 180, 0, 256]

            hist1 = cv2.calcHist([hsv1], [0, 1], None, histSize, ranges)
            hist2 = cv2.calcHist([hsv2], [0, 1], None, histSize, ranges)

            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

            color_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            # ORB feature matching
            gray1 = cv2.cvtColor(snapshot_img, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(owner_img, cv2.COLOR_BGR2GRAY)

            max_dim = 512
            scale1 = max_dim / max(gray1.shape)
            scale2 = max_dim / max(gray2.shape)

            if scale1 < 1:
                gray1 = cv2.resize(gray1, (int(gray1.shape[1] * scale1), int(gray1.shape[0] * scale1)))
            if scale2 < 1:
                gray2 = cv2.resize(gray2, (int(gray2.shape[1] * scale2), int(gray2.shape[0] * scale2)))

            keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
            keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

            feature_confidence, feature_sim, match_points = 0, 0, 0
            if descriptors1 is not None and descriptors2 is not None and len(descriptors1) > 10 and len(descriptors2) > 10:
                matches = bf.match(descriptors1, descriptors2)
                matches = sorted(matches, key=lambda x: x.distance)

                good_matches = [m for m in matches[:30] if m.distance < FEATURE_MATCH_DISTANCE_THRESHOLD]

                if len(good_matches) > 0:
                    avg_distance = sum(m.distance for m in good_matches) / len(good_matches)
                    feature_confidence = 1.0 - (avg_distance / FEATURE_MATCH_DISTANCE_THRESHOLD)
                    match_ratio = len(good_matches) / min(len(keypoints1), len(keypoints2))
                    feature_sim = match_ratio
                    match_points = len(good_matches)

            # Combined similarity score
            if feature_confidence > FEATURE_CONFIDENCE_THRESHOLD:
                combined_sim = 0.4 * color_sim + 0.6 * feature_sim
                method = "feature_strong"
            else:
                combined_sim = 0.7 * color_sim + 0.3 * feature_sim
                method = "color_strong"

            if color_sim >= COLOR_SIMILARITY_THRESHOLD or feature_confidence > 0.5:
                match_info = {
                    'filename': fname,
                    'combined_score': combined_sim,
                    'color_score': color_sim,
                    'feature_score': feature_sim,
                    'feature_confidence': feature_confidence,
                    'match_points': match_points,
                    'method': method,
                    'path': path,
                    'owner_id': owner_id
                }
                all_matches.append(match_info)
                visual_matches.append((fname, combined_sim, method))

            if combined_sim > best_score and combined_sim >= threshold:
                best_score = combined_sim
                best_match = fname
                match_method = method
                match_owner_id = owner_id

        except Exception as e:
            print(f"Error comparing images: {e}")

    # Fallback: CNN embedding
    if best_match is None:
        query_embedding = get_image_embedding(snapshot_img)

        for fname, data in owner_embeddings.items():
            db_embedding = data.get("embedding")
            owner_id = data.get("owner_id")

            sim = cosine_similarity([query_embedding], [db_embedding])[0][0]
            embedding_matches.append((fname, sim, "embedding"))

            if sim > best_score and sim >= threshold * 0.9:
                best_score = sim
                best_match = fname
                match_method = "embedding"
                match_owner_id = owner_id

                all_matches.append({
                    'filename': fname,
                    'combined_score': sim,
                    'color_score': 0,
                    'feature_score': 0,
                    'feature_confidence': 0,
                    'match_points': 0,
                    'method': 'embedding',
                    'path': os.path.join(DATABASE_PATH, owner_id, fname),
                    'owner_id': owner_id
                })

    all_method_matches = visual_matches + embedding_matches
    all_method_matches.sort(key=lambda x: x[1], reverse=True)

    if best_match:
        print(f"Matched to owner image '{best_match}' (owner: {match_owner_id}) using {match_method} score {best_score:.2f}")

    return {
        'match': best_match,
        'owner_id': match_owner_id,
        'score': float(best_score) if best_score > -1 else 0.0,
        'method': match_method,
        'all_matches': all_matches,
        'highest_confidence_matches': all_method_matches[:5] if all_method_matches else []
    } if best_match else {
        'all_matches': all_matches,
        'highest_confidence_matches': all_method_matches[:5] if all_method_matches else []
    } if all_matches else None

def calculate_appearance_similarity(img1, img2):
    """
    Calculate visual similarity between two animal crops
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Resize images to the same size for comparison
    size = (64, 64)
    img1_resized = cv2.resize(img1, size)
    img2_resized = cv2.resize(img2, size)
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Use a combination of histogram comparison
    hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # Method 2: Use feature matching with ORB
    try:
        # Setup ORB feature detector
        orb = cv2.ORB_create(nfeatures=10000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Detect ORB keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        
        # Check if enough keypoints were found
        if des1 is not None and des2 is not None and len(des1) > 10 and len(des2) > 10:
            # Match descriptors
            matches = bf.match(des1, des2)
            
            # Calculate feature similarity based on number of good matches
            feature_similarity = len(matches) / max(len(kp1), len(kp2))
            
            # Combine both methods, giving more weight to feature matching
            combined_similarity = 0.3 * hist_similarity + 0.7 * feature_similarity
        else:
            # Fall back to histogram similarity if not enough keypoints
            combined_similarity = hist_similarity
    except:
        # If ORB fails, just use histogram similarity
        combined_similarity = hist_similarity
    
    return combined_similarity

def find_best_match(snapshot_img):
    """
    Find the best match for a snapshot image among registered pets
    
    Args:
        snapshot_img: Grayscale image of the detected animal
        
    Returns:
        str: Owner ID if a match is found, None otherwise
    """
    # This is a simplified version for compatibility
    match_result = match_snapshot_to_owner(snapshot_img)
    if match_result and match_result.get('owner_id'):
        return match_result.get('owner_id')
    return None