#!/usr/bin/env python
# notification.py - Notification system for animal detections

import time
from datetime import datetime

# Import configuration
from config.constants import MAX_NOTIFICATION_HISTORY
from data.database import notification_history

def notify_owner(owner_id, animal_img=None, animal_info=None):
    """
    Notify the owner of a detected pet
    
    Args:
        owner_id: The owner's ID or image filename
        animal_img: The detected animal image
        animal_info: Additional information about the detection
        
    Returns:
        dict: Notification information
    """
    timestamp = datetime.now().isoformat()
    stream_id = animal_info.get('stream_id', 'unknown') if animal_info else 'unknown'
    
    notification = {
        "id": f"owner_notify_{int(time.time())}",
        "type": "owner_notification",
        "owner_id": owner_id,
        "timestamp": timestamp,
        "status": "sent",
        "animal_info": animal_info,
        "stream_id": stream_id  # Explicitly include stream_id in notification
    }
    
    # Save the notification to history
    notification_history.insert(0, notification)
    
    # Limit notification history size
    if len(notification_history) > MAX_NOTIFICATION_HISTORY:
        notification_history.pop()
    
    print(f"Owner notified for animal ID: {owner_id} from stream: {stream_id}")
    return notification

def notify_pound(image_path, animal_info=None):
    """
    Notify the animal pound about a stray animal
    
    Args:
        image_path: Path to the animal image
        animal_info: Additional information about the detection
        
    Returns:
        dict: Notification information
    """
    timestamp = datetime.now().isoformat()
    stream_id = animal_info.get('stream_id', 'unknown') if animal_info else 'unknown'
    
    notification = {
        "id": f"pound_notify_{int(time.time())}",
        "type": "pound_notification",
        "image_path": image_path,
        "timestamp": timestamp,
        "status": "sent",
        "animal_info": animal_info,
        "stream_id": stream_id  # Explicitly include stream_id in notification
    }
    
    # Save the notification to history
    notification_history.insert(0, notification)
    
    # Limit notification history size
    if len(notification_history) > MAX_NOTIFICATION_HISTORY:
        notification_history.pop()
    
    print(f"Animal pound notified with image: {image_path} from stream: {stream_id}")
    return notification

def get_notifications(filters=None, limit=50):
    """
    Get notifications with optional filtering
    
    Args:
        filters: Dictionary of filter criteria
        limit: Maximum number of notifications to return
        
    Returns:
        list: Filtered notifications
    """
    if filters is None:
        filters = {}
    
    filtered_notifications = notification_history.copy()
    
    # Apply filters
    if 'type' in filters:
        filtered_notifications = [n for n in filtered_notifications if n.get('type') == filters['type']]
    
    if 'stream_id' in filters:
        # Check both the direct stream_id and the one in animal_info
        filtered_notifications = [n for n in filtered_notifications if 
                               (n.get('stream_id') == filters['stream_id'] or 
                                n.get('animal_info', {}).get('stream_id') == filters['stream_id'])]
    
    if 'animal_type' in filters:
        filtered_notifications = [n for n in filtered_notifications if 
                                n.get('animal_info', {}).get('animal_type') == filters['animal_type']]
        
    if 'case' in filters:
        filtered_notifications = [n for n in filtered_notifications if 
                                n.get('animal_info', {}).get('notification_case') == filters['case']]
        
    if 'owner_id' in filters:
        filtered_notifications = [n for n in filtered_notifications if 
                                n.get('owner_id') == filters['owner_id']]
    
    # Limit the number of results
    return filtered_notifications[:limit]

def get_notification_by_id(notification_id):
    """
    Get a notification by its ID
    
    Args:
        notification_id: ID of the notification
        
    Returns:
        dict: Notification data or None if not found
    """
    for notification in notification_history:
        if notification.get('id') == notification_id:
            return notification
    return None

def generate_notification_stats():
    """
    Generate statistics about notifications
    
    Returns:
        dict: Notification statistics
    """
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
    
    return {
        "total": len(notification_history),
        "by_type": by_type,
        "by_case": by_case,
        "by_stream": by_stream,
        "by_animal_type": by_animal_type,
        "recent_streams": list(by_stream.keys())[:5]  # Include 5 most recent streams for quick reference
    }