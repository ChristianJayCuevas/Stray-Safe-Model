#!/usr/bin/env python
# feature_extractor.py - Image feature extraction for matching

import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input

# Global model instance
feature_extractor = None

def initialize_feature_extractor():
    """Initialize the ResNet50 feature extractor"""
    global feature_extractor
    
    print("Initializing ResNet50 feature extractor...")
    try:
        # Load ResNet50 model
        feature_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        print("Feature extractor initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing feature extractor: {e}")
        return False

def extract_features(image):
    """
    Extract features from an image
    
    Args:
        image: RGB image (numpy array)
        
    Returns:
        numpy.ndarray: Feature vector
    """
    global feature_extractor
    
    if feature_extractor is None:
        initialize_feature_extractor()
    
    # Ensure image is in the correct format
    from tensorflow.keras.applications.resnet50 import preprocess_input
    import numpy as np
    
    # Resize to expected shape
    # Convert to RGB if needed
    import cv2
    if len(image.shape) == 3 and image.shape[2] == 3:
        if image.dtype != np.float32:
            # Check if BGR (OpenCV) instead of RGB
            if cv2 is not None:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Simple channel swap if cv2 is not available
                rgb_image = image[..., ::-1]
        else:
            rgb_image = image
    else:
        raise ValueError("Input image must be a 3-channel color image")
    
    # Resize to 224x224 if not already
    if rgb_image.shape[0] != 224 or rgb_image.shape[1] != 224:
        rgb_image = cv2.resize(rgb_image, (224, 224))
    
    # Expand dimensions and preprocess
    x = np.expand_dims(rgb_image, axis=0)
    x = preprocess_input(x)
    
    # Extract features
    features = feature_extractor.predict(x)
    
    return features[0]  # Return the feature vector