#!/usr/bin/env python
# classifier.py - Animal classification models

import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
from torchvision import models, transforms
from config.constants import MODEL_DIRECTORY, CNN_MODEL_PATH

# Global model instances
cnn_model = None
vgg_model = None

# Image transformation for VGG model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def initialize_classifier():
    """Initialize the CNN model for stray/not-stray classification"""
    global cnn_model
    
    print(f"Loading classifier model from {CNN_MODEL_PATH}...")
    try:
        # Load the model
        cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
        print("Classifier model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading classifier model: {e}")
        return False

def initialize_vgg16():
    """Initialize VGG16 model for animal classification (dog/cat)"""
    global vgg_model
    
    print(f"Loading VGG16 model from {MODEL_DIRECTORY}...")
    try:
        # Create VGG16 model with batch normalization
        vgg16 = models.vgg16_bn(weights=None)  # No downloading; using .pth

        # Freeze feature layers
        for param in vgg16.features.parameters():
            param.requires_grad = False

        # Modify classifier
        num_features = vgg16.classifier[-1].in_features
        features = list(vgg16.classifier.children())[:-1]
        features.extend([nn.Linear(num_features, 2)])
        vgg16.classifier = nn.Sequential(*features)

        # Load custom trained weights
        vgg16.load_state_dict(torch.load(MODEL_DIRECTORY, map_location=torch.device('cpu')))
        vgg16.eval()
        
        vgg_model = vgg16
        print("VGG16 model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading VGG16 model: {e}")
        return False

def preprocess_image(image):
    """
    Preprocess an image for the CNN model
    
    Args:
        image: OpenCV image
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    img = cv2.resize(image, (128, 128))
    img = np.expand_dims(img, axis=-1) if len(img.shape) == 2 else img
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def classify_image(image):
    """
    Classify an image as stray or not stray
    
    Args:
        image: OpenCV image
        
    Returns:
        tuple: (classification_result, confidence)
    """
    global cnn_model
    
    if cnn_model is None:
        initialize_classifier()
        
    # Preprocess the image
    preprocessed = preprocess_image(image)
    
    # Make prediction
    prediction = cnn_model.predict(preprocessed)[0]
    
    # Determine classification
    is_stray = prediction >= 0.3
    classification = "stray" if is_stray else "not_stray"
    
    return classification, float(prediction)

def classify_animal_type(image_pil):
    """
    Classify an image as dog or cat
    
    Args:
        image_pil: PIL image
        
    Returns:
        tuple: (animal_type, confidence)
    """
    global vgg_model
    
    if vgg_model is None:
        initialize_vgg16()
        
    # Transform the PIL image for the model
    tensor = transform(image_pil).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        outputs = vgg_model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probs, 1)

    # Map prediction to label
    animal_type = "cat" if prediction.item() == 0 else "dog"
    
    return animal_type, confidence.item()