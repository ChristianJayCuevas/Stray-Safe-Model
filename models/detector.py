#!/usr/bin/env python
# detector.py - Object detection model for animal detection

import torch
from ultralytics import RTDETR
from config.constants import DEVICE, YOLO_MODEL_PATH

# Global model instance
model = None

def initialize_detector():
    """Initialize the RTDETR model for object detection"""
    global model
    
    print(f"Initializing object detection model from {YOLO_MODEL_PATH}...")
    try:
        # Load the RTDETR model
        model = RTDETR(YOLO_MODEL_PATH)
        
        # Set detection confidence threshold
        model.conf = 0.6
        
        # Move model to appropriate device (CUDA or CPU)
        model.to(DEVICE)
        
        print(f"Detection model loaded successfully on {DEVICE}")
        return True
    except Exception as e:
        print(f"Error initializing detector model: {e}")
        return False

def detect_objects(image, conf=0.6, classes=None):
    """
    Detect objects in an image
    
    Args:
        image: Input image (numpy array)
        conf: Confidence threshold (default: 0.6)
        classes: List of class IDs to detect (default: all)
        
    Returns:
        list: Detected objects with bounding boxes and classes
    """
    global model
    
    if model is None:
        initialize_detector()
        
    # Set confidence threshold
    model.conf = conf
    
    # Run inference
    results = model.predict(image, device=DEVICE)
    
    # Process results
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = box.conf.item()
            
            # Filter by classes if specified
            if classes is not None and class_id not in classes:
                continue
                
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Store detection information
            detection = {
                "class_id": class_id,
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                "area": (x2 - x1) * (y2 - y1),
                "width": x2 - x1,
                "height": y2 - y1
            }
            
            detections.append(detection)
    
    return detections