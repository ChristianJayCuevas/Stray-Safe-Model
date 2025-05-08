#!/usr/bin/env python
# routes.py - Flask API routes definition

from flask import request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import time
from datetime import datetime

# Import API handlers
from api.handlers import (
    handle_detected_animals,
    handle_detected_image,
    handle_animal_stats,
    handle_debug_pipeline,
    handle_debug_image,
    handle_streams_list,
    handle_animal_counters,
    handle_video_snapshot,
    handle_hls_file,
    handle_predict_animal,
    handle_all_matches,
    handle_notification_list,
    handle_notification_stats,
    handle_notification_details,
    handle_database_status,
    handle_test_notification,
    handle_upload_pet_image,
    handle_get_pet_image,
    handle_remote_recorders_status,
    handle_toggle_remote_recorders,
    handle_remote_recorder_snapshot,
    handle_breed_classification
)

def setup_routes(app):
    """Set up all API routes for the Flask application"""
    
    # --- Get detected animals ---
    @app.route('/api2/detected', methods=['GET'])
    def get_detected_animals():
        return handle_detected_animals(request)
    
    # --- Serve detected animal images ---
    @app.route('/api2/detected-img/<stream_id>/<filename>')
    def serve_detected_image(stream_id, filename):
        return handle_detected_image(stream_id, filename)
    
    # --- Get statistics about detected animals ---
    @app.route('/api2/stats')
    def get_animal_stats():
        return handle_animal_stats()
    
    # --- Debug detection pipeline for a stream ---
    @app.route('/api2/debug/<stream_id>')
    def debug_pipeline(stream_id):
        return handle_debug_pipeline(stream_id)
    
    # --- Serve debug images ---
    @app.route('/api2/debug-img/<stream_id>/<path:filename>')
    def serve_debug_image(stream_id, filename):
        return handle_debug_image(stream_id, filename)
    
    # --- Get information about active streams ---
    @app.route('/api2/streams')
    def get_all_streams():
        return handle_streams_list()
    
    # --- Get animal counters for each stream ---
    @app.route('/api2/counters')
    def get_animal_counters():
        return handle_animal_counters()
    
    # --- Get a snapshot of the current video frame ---
    @app.route('/api2/video/<stream_id>')
    def video_snapshot(stream_id):
        return handle_video_snapshot(stream_id)
    
    # --- Serve HLS stream files ---
    @app.route('/api2/hls/<stream_id>/<path:filename>')
    def serve_hls_file(stream_id, filename):
        return handle_hls_file(stream_id, filename)
    
    # --- Predict if an animal is a stray ---
    @app.route("/api2/predict", methods=["POST"])
    def predict():
        return handle_predict_animal(request)
    
    # --- Show all potential matches for an animal ---
    @app.route('/api2/all-matches/<stream_id>')
    def show_all_matches(stream_id):
        return handle_all_matches(stream_id)
    
    # --- Get notifications ---
    @app.route('/api2/notifications', methods=['GET'])
    def get_notifications():
        return handle_notification_list(request)
    
    # --- Get notification statistics ---
    @app.route('/api2/notifications/stats', methods=['GET'])
    def get_notification_stats():
        return handle_notification_stats()
    
    # --- Get details for a specific notification ---
    @app.route('/api2/notifications/<notification_id>', methods=['GET'])
    def get_notification_details(notification_id):
        return handle_notification_details(notification_id)
    
    # --- Check database status ---
    @app.route('/api2/database/status')
    def check_database():
        return handle_database_status()
    
    # --- Create a test notification ---
    @app.route('/api2/test-notification', methods=['GET'])
    def test_notification():
        return handle_test_notification(request)
    
    # --- Upload a pet image ---
    @app.route('/api2/upload-pet-image', methods=['POST'])
    def upload_pet_image():
        return handle_upload_pet_image(request)
    
    # --- Get a pet image ---
    @app.route('/api2/pet-image/<username>/<filename>', methods=['GET'])
    def get_pet_image(username, filename):
        return handle_get_pet_image(username, filename)
    
    # --- Get status of remote recorders ---
    @app.route('/api2/remote-recorders', methods=['GET'])
    def get_remote_recorders_status():
        return handle_remote_recorders_status(request)
    
    # --- Toggle remote recorders ---
    @app.route('/api2/remote-recorders/toggle', methods=['POST'])
    def toggle_remote_recorders():
        return handle_toggle_remote_recorders(request)
    
    # --- Get a snapshot from a remote recorder ---
    @app.route('/api2/remote-recorders/<stream_id>/snapshot', methods=['GET'])
    def get_remote_recorder_snapshot(stream_id):
        return handle_remote_recorder_snapshot(stream_id)
    
    # --- Classify breed from image ---
    @app.route('/api2/classify-breed', methods=['POST'])
    def classify_breed():
        return handle_breed_classification(request)
    
    return app