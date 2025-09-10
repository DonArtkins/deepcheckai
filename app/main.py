import os
import uuid
import threading
from flask import request, jsonify
from flask_socketio import emit
from app import socketio, create_app
from app.utils.file_processing import download_file, cleanup_file
from app.utils.helpers import validate_api_key
from app.models.image_model import analyze_image
from app.models.video_model import analyze_video
from app.models.audio_model import analyze_audio

app = create_app()

# Store analysis progress
analysis_progress = {}

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_analysis')
def handle_start_analysis(data):
    session_id = request.sid
    media_url = data.get('media_url')
    media_type = data.get('media_type')
    
    # Start analysis in background thread
    thread = threading.Thread(
        target=process_media, 
        args=(session_id, media_url, media_type)
    )
    thread.daemon = True
    thread.start()
    
    emit('analysis_started', {'status': 'processing', 'message': 'Analysis started'})

def process_media(session_id, media_url, media_type):
    try:
        # Download file
        file_path = download_file(media_url)
        
        if not file_path:
            emit('analysis_error', {'error': 'Failed to download file'}, room=session_id)
            return
        
        # Update progress
        analysis_progress[session_id] = 10
        emit('progress_update', {'progress': 10}, room=session_id)
        
        # Analyze based on media type
        if media_type in ['image', 'jpg', 'jpeg', 'png']:
            result = analyze_image(file_path, session_id, analysis_progress)
        elif media_type in ['video', 'mp4', 'mov', 'avi', 'mkv']:
            result = analyze_video(file_path, session_id, analysis_progress)
        elif media_type in ['audio', 'wav', 'mp3']:
            result = analyze_audio(file_path, session_id, analysis_progress)
        else:
            raise ValueError(f"Unsupported media type: {media_type}")
        
        # Cleanup
        cleanup_file(file_path)
        
        # Send final result
        emit('analysis_complete', result, room=session_id)
        
        # Remove progress tracking
        if session_id in analysis_progress:
            del analysis_progress[session_id]
            
    except Exception as e:
        print(f"Error processing media: {str(e)}")
        emit('analysis_error', {'error': str(e)}, room=session_id)
        
        # Cleanup on error
        if 'file_path' in locals() and os.path.exists(file_path):
            cleanup_file(file_path)
        
        if session_id in analysis_progress:
            del analysis_progress[session_id]

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=8000)