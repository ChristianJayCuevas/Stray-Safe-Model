#!/usr/bin/env python
# storage.py - File storage operations

import os
import shutil
import glob
import time

# Import configuration
from config.constants import UPLOAD_BASE_DIR, DATABASE_PATH, HLS_CLEANUP_DIR

def setup_storage():
    """
    Set up storage directories
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create necessary directories
        os.makedirs(UPLOAD_BASE_DIR, exist_ok=True)
        os.makedirs(DATABASE_PATH, exist_ok=True)
        os.makedirs(HLS_CLEANUP_DIR, exist_ok=True)
        
        # Set permissions for HLS directory
        os.chmod(HLS_CLEANUP_DIR, 0o777)
        
        # Create debug directories
        debug_dir = os.path.join("venv", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        print("Storage directories set up successfully")
        return True
    except Exception as e:
        print(f"Error setting up storage: {e}")
        return False

def cleanup_old_recordings(directory, max_size_gb=10, max_files=100, min_free_space_gb=2):
    """
    Clean up old recording files when disk space is low or too many files
    
    Args:
        directory: The directory containing recording files
        max_size_gb: Maximum size of recordings in GB
        max_files: Maximum number of files to keep
        min_free_space_gb: Minimum free space to maintain in GB
    
    Returns:
        int: Number of files deleted
    """
    try:
        # Get all video files in the directory (mp4, avi, etc.)
        video_files = []
        for ext in ['.mp4', '.avi', '.mkv', '.mov', '.wmv']:
            video_files.extend(glob.glob(os.path.join(directory, f'*{ext}')))
        
        # If there are no files, return
        if not video_files:
            return 0
        
        # Sort files by modification time (oldest first)
        video_files.sort(key=os.path.getmtime)
        
        # Check if we have too many files
        files_to_delete = []
        if len(video_files) > max_files:
            files_to_delete.extend(video_files[0:len(video_files) - max_files])
        
        # Check total size of recordings
        total_size_gb = sum(os.path.getsize(f) for f in video_files) / (1024**3)
        
        # If total size exceeds max_size_gb, delete oldest files
        if total_size_gb > max_size_gb:
            # Calculate how many GB we need to remove
            size_to_remove_gb = total_size_gb - max_size_gb
            size_removed_gb = 0
            
            for file in video_files:
                if file not in files_to_delete:
                    file_size_gb = os.path.getsize(file) / (1024**3)
                    size_removed_gb += file_size_gb
                    files_to_delete.append(file)
                    
                    if size_removed_gb >= size_to_remove_gb:
                        break
        
        # Check available disk space
        try:
            import psutil
            disk_usage = psutil.disk_usage(os.path.dirname(directory))
            free_space_gb = disk_usage.free / (1024**3)
            
            if free_space_gb < min_free_space_gb:
                # Calculate how many GB we need to free up
                space_to_free_gb = min_free_space_gb - free_space_gb
                space_freed_gb = 0
                
                for file in video_files:
                    if file not in files_to_delete:
                        file_size_gb = os.path.getsize(file) / (1024**3)
                        space_freed_gb += file_size_gb
                        files_to_delete.append(file)
                        
                        if space_freed_gb >= space_to_free_gb:
                            break
        except ImportError:
            print("psutil not available, skipping disk space check")
        
        # Delete the files
        deleted_count = 0
        for file in files_to_delete:
            try:
                os.remove(file)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting file {file}: {e}")
        
        return deleted_count
    
    except Exception as e:
        print(f"Error in cleanup_old_recordings: {e}")
        return 0

def cleanup_hls_segments(max_age_hours=24):
    """
    Clean up old HLS segments
    
    Args:
        max_age_hours: Maximum age of HLS segments in hours
        
    Returns:
        int: Number of files deleted
    """
    try:
        # Get all HLS segment files
        segment_files = glob.glob(os.path.join(HLS_CLEANUP_DIR, '*.ts'))
        
        # Get current time
        current_time = time.time()
        
        # Calculate cutoff time
        cutoff_time = current_time - (max_age_hours * 3600)
        
        # Delete old files
        deleted_count = 0
        for file in segment_files:
            if os.path.getmtime(file) < cutoff_time:
                try:
                    os.remove(file)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting HLS segment {file}: {e}")
        
        return deleted_count
    except Exception as e:
        print(f"Error cleaning up HLS segments: {e}")
        return 0

def save_pet_image(file_data, owner_id, filename):
    """
    Save a pet image to the database
    
    Args:
        file_data: Binary file data
        owner_id: Owner ID
        filename: Filename to save as
        
    Returns:
        str: Path to the saved file
    """
    try:
        # Create owner directory if it doesn't exist
        owner_dir = os.path.join(DATABASE_PATH, owner_id)
        os.makedirs(owner_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(owner_dir, filename)
        with open(file_path, 'wb') as f:
            f.write(file_data)
        
        return file_path
    except Exception as e:
        print(f"Error saving pet image: {e}")
        return None

def get_pet_image_path(owner_id, filename):
    """
    Get path to a pet image
    
    Args:
        owner_id: Owner ID
        filename: Filename
        
    Returns:
        str: Path to the file
    """
    return os.path.join(DATABASE_PATH, owner_id, filename)