#!/usr/bin/env python
# breed_classifier.py - Breed classification model based on EfficientNetB0

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Import configuration
from config.constants import EFFICIENTNET_MODEL_PATH, DOG_BREEDS, CAT_BREEDS

# Global model instance
breed_model = None

def initialize_breed_classifier():
    """Initialize the EfficientNetB0 model for breed classification"""
    global breed_model
    
    print(f"Loading breed classifier model from {EFFICIENTNET_MODEL_PATH}...")
    try:
        # Load the model
        breed_model = load_model(EFFICIENTNET_MODEL_PATH)
        print("Breed classifier model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading breed classifier model: {e}")
        return False

def preprocess_image_for_breed(image):
    """
    Preprocess an image for the breed classifier model
    
    Args:
        image: OpenCV image (BGR format)
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    img = cv2.resize(image_rgb, (224, 224))
    
    # Convert to float and expand dimensions
    img = np.expand_dims(img, axis=0)
    
    # Preprocess for EfficientNet
    img = preprocess_input(img)
    
    return img

def predict_breed(image, animal_type):
    """
    Predict the breed of a dog or cat
    
    Args:
        image: OpenCV image
        animal_type: 'dog' or 'cat'
        
    Returns:
        tuple: (breed_name, confidence)
    """
    global breed_model
    
    if breed_model is None:
        initialize_breed_classifier()
    
    # Preprocess the image
    processed_img = preprocess_image_for_breed(image)
    
    # Make prediction
    predictions = breed_model.predict(processed_img)[0]
    
    # Get the index of the highest probability class
    predicted_class_idx = np.argmax(predictions)
    confidence = float(predictions[predicted_class_idx])
    
    # Map index to breed name based on animal type
    if animal_type == 'dog':
        breed_list = DOG_BREEDS
    else:  # cat
        breed_list = CAT_BREEDS
    
    # Make sure the index is valid
    if predicted_class_idx < len(breed_list):
        breed_name = breed_list[predicted_class_idx]
    else:
        breed_name = "Unknown"
    
    return breed_name, confidence 