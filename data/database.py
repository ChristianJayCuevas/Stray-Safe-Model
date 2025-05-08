#!/usr/bin/env python
# database.py - Database operations for storing and retrieving data

from collections import defaultdict

# --- Global variables to store application data ---

# Store stream data for each stream
stream_data = {}

# Store detected animals
detected_animals_log = []

# Store notification history
notification_history = []

# Store animal counters for each stream
animal_counters = defaultdict(lambda: {"dog": 0, "cat": 0})

# Store owner embeddings for pet matching
owner_embeddings = {}

def initialize_database():
    """Initialize database connections and structures"""
    # For now, we're using in-memory data structures
    # This could be expanded to use a real database in the future
    print("Initializing database...")
    
    # Initialize global variables
    global stream_data, detected_animals_log, notification_history, animal_counters, owner_embeddings
    
    # Make sure they're empty/initialized
    stream_data = {}
    detected_animals_log = []
    notification_history = []
    animal_counters = defaultdict(lambda: {"dog": 0, "cat": 0})
    owner_embeddings = {}
    
    print("Database initialized successfully")
    return True

def get_stream_data(stream_id=None):
    """
    Get stream data for a specific stream or all streams
    
    Args:
        stream_id: Optional stream ID to get data for
        
    Returns:
        dict: Stream data
    """
    if stream_id:
        return stream_data.get(stream_id, {})
    return stream_data

def get_detected_animals(limit=100, **filters):
    """
    Get detected animals with optional filtering
    
    Args:
        limit: Maximum number of animals to return
        **filters: Filter criteria
        
    Returns:
        list: Filtered animals
    """
    filtered = detected_animals_log.copy()
    
    # Apply filters
    for key, value in filters.items():
        filtered = [a for a in filtered if a.get(key) == value]
    
    # Apply limit
    return filtered[:limit]

def get_notifications(limit=100, **filters):
    """
    Get notifications with optional filtering
    
    Args:
        limit: Maximum number of notifications to return
        **filters: Filter criteria
        
    Returns:
        list: Filtered notifications
    """
    filtered = notification_history.copy()
    
    # Apply filters
    for key, value in filters.items():
        filtered = [n for n in filtered if n.get(key) == value]
    
    # Apply limit
    return filtered[:limit]

def get_animal_counters(stream_id=None):
    """
    Get animal counters for a specific stream or all streams
    
    Args:
        stream_id: Optional stream ID to get counters for
        
    Returns:
        dict: Animal counters
    """
    if stream_id:
        return animal_counters.get(stream_id, {"dog": 0, "cat": 0})
    return animal_counters

def get_owner_embeddings():
    """
    Get all owner embeddings
    
    Returns:
        dict: Owner embeddings
    """
    return owner_embeddings

def get_owner_pets(owner_id):
    """
    Get all pets registered to an owner
    
    Args:
        owner_id: Owner ID
        
    Returns:
        list: Pet information
    """
    pets = []
    for filename, data in owner_embeddings.items():
        if data.get('owner_id') == owner_id:
            pets.append({
                'filename': filename,
                'embedding': data.get('embedding'),
                'owner_id': owner_id
            })
    return pets

def register_pet(owner_id, filename, embedding):
    """
    Register a new pet with its embedding
    
    Args:
        owner_id: Owner ID
        filename: Pet image filename
        embedding: Pet feature embedding
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        owner_embeddings[filename] = {
            'embedding': embedding,
            'owner_id': owner_id
        }
        return True
    except Exception as e:
        print(f"Error registering pet: {e}")
        return False

def add_detection(detection):
    """
    Add a new animal detection to the log
    
    Args:
        detection: Detection data
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        detected_animals_log.insert(0, detection)
        # Keep the log at a reasonable size
        if len(detected_animals_log) > 1000:
            detected_animals_log.pop()
        return True
    except Exception as e:
        print(f"Error adding detection: {e}")
        return False

def add_notification(notification):
    """
    Add a new notification to the history
    
    Args:
        notification: Notification data
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        notification_history.insert(0, notification)
        # Keep the history at a reasonable size
        if len(notification_history) > 1000:
            notification_history.pop()
        return True
    except Exception as e:
        print(f"Error adding notification: {e}")
        return False