import os
import uuid
import threading
import logging
from flask import request, jsonify
from flask_socketio import emit
from app import socketio, create_app
from app.utils.file_processing import download_file, cleanup_file
from app.utils.helpers import validate_api_key

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = create_app()

# Store analysis progress
analysis_progress = {}

# Model initialization flag
models_initialized = False

def initialize_models():
    """Initialize all ML models at startup"""
    global models_initialized
    
    try:
        logger.info("Initializing deepfake detection models...")
        
        # Import models to trigger initialization
        from app.models.image_model import detector as image_detector
        from app.models.video_model import detector as video_detector
        from app.models.audio_model import detector as audio_detector
        
        logger.info("Image model initialized: " + str(image_detector.model is not None))
        logger.info("Video model initialized: " + str(video_detector.model is not None))
        logger.info("Audio model initialized: " + str(audio_detector.model is not None))
        
        models_initialized = True
        logger.info("All models initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        models_initialized = False

@socketio.on('connect')
def handle_connect():
    logger.info(f'Client connected: {request.sid}')
    emit('connection_status', {
        'status': 'connected',
        'models_ready': models_initialized,
        'message': 'Connected to deepfake detection service'
    })

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f'Client disconnected: {request.sid}')
    # Clean up any ongoing analysis for this session
    session_id = request.sid
    if session_id in analysis_progress:
        del analysis_progress[session_id]

@socketio.on('start_analysis')
def handle_start_analysis(data):
    session_id = request.sid
    media_url = data.get('media_url')
    media_type = data.get('media_type')
    
    if not media_url or not media_type:
        emit('analysis_error', {
            'error': 'Missing media_url or media_type',
            'code': 'MISSING_PARAMETERS'
        })
        return
    
    if not models_initialized:
        emit('analysis_error', {
            'error': 'Models not initialized yet. Please try again in a moment.',
            'code': 'MODELS_NOT_READY'
        })
        return
    
    logger.info(f"Starting analysis for session {session_id}: {media_type} - {media_url}")
    
    # Start analysis in background thread
    thread = threading.Thread(
        target=process_media, 
        args=(session_id, media_url, media_type),
        daemon=True
    )
    thread.start()
    
    emit('analysis_started', {
        'status': 'processing', 
        'message': f'Analysis started for {media_type}',
        'session_id': session_id
    })

def process_media(session_id, media_url, media_type):
    """Process media file for deepfake detection"""
    file_path = None
    
    try:
        logger.info(f"Processing {media_type} for session {session_id}")
        
        # Update progress
        analysis_progress[session_id] = 5
        emit('progress_update', {'progress': 5, 'message': 'Downloading file...'}, room=session_id)
        
        # Download file
        file_path = download_file(media_url)
        
        if not file_path:
            emit('analysis_error', {
                'error': 'Failed to download file. Please check the URL.',
                'code': 'DOWNLOAD_FAILED'
            }, room=session_id)
            return
        
        # Verify file exists and has content
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            emit('analysis_error', {
                'error': 'Downloaded file is empty or corrupted',
                'code': 'FILE_CORRUPTED'
            }, room=session_id)
            return
        
        # Update progress
        analysis_progress[session_id] = 10
        emit('progress_update', {'progress': 10, 'message': 'File downloaded, starting analysis...'}, room=session_id)
        
        # Import models dynamically to avoid circular imports
        result = None
        
        # Analyze based on media type
        if media_type.lower() in ['image', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']:
            from app.models.image_model import analyze_image
            result = analyze_image(file_path, session_id, analysis_progress)
            
        elif media_type.lower() in ['video', 'mp4', 'mov', 'avi', 'mkv', 'webm', 'flv']:
            from app.models.video_model import analyze_video
            result = analyze_video(file_path, session_id, analysis_progress)
            
        elif media_type.lower() in ['audio', 'wav', 'mp3', 'flac', 'aac', 'ogg', 'm4a']:
            from app.models.audio_model import analyze_audio
            result = analyze_audio(file_path, session_id, analysis_progress)
            
        else:
            emit('analysis_error', {
                'error': f"Unsupported media type: {media_type}",
                'code': 'UNSUPPORTED_MEDIA_TYPE',
                'supported_types': {
                    'image': ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff'],
                    'video': ['mp4', 'mov', 'avi', 'mkv', 'webm', 'flv'],
                    'audio': ['wav', 'mp3', 'flac', 'aac', 'ogg', 'm4a']
                }
            }, room=session_id)
            return
        
        if result is None:
            emit('analysis_error', {
                'error': 'Analysis failed to produce results',
                'code': 'ANALYSIS_FAILED'
            }, room=session_id)
            return
        
        # Add metadata to result
        result['session_id'] = session_id
        result['media_url'] = media_url
        result['file_size'] = os.path.getsize(file_path)
        
        # Send final result
        emit('analysis_complete', result, room=session_id)
        logger.info(f"Analysis completed for session {session_id}: {result.get('isDeepfake', 'unknown')} with {result.get('confidence', 0):.1f}% confidence")
        
    except Exception as e:
        logger.error(f"Error processing media for session {session_id}: {str(e)}")
        emit('analysis_error', {
            'error': str(e),
            'code': 'PROCESSING_ERROR'
        }, room=session_id)
        
    finally:
        # Cleanup
        if file_path and os.path.exists(file_path):
            cleanup_file(file_path)
        
        # Remove progress tracking
        if session_id in analysis_progress:
            del analysis_progress[session_id]

@socketio.on('get_progress')
def handle_get_progress(data):
    """Get current analysis progress"""
    session_id = data.get('session_id', request.sid)
    progress = analysis_progress.get(session_id, 0)
    emit('progress_update', {'progress': progress})

@socketio.on('cancel_analysis')
def handle_cancel_analysis(data):
    """Cancel ongoing analysis"""
    session_id = data.get('session_id', request.sid)
    
    if session_id in analysis_progress:
        del analysis_progress[session_id]
        emit('analysis_cancelled', {'message': 'Analysis cancelled'})
        logger.info(f"Analysis cancelled for session {session_id}")
    else:
        emit('analysis_error', {'error': 'No ongoing analysis found'})

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get service status"""
    return jsonify({
        'status': 'healthy',
        'service': 'deepfake-ai',
        'models_initialized': models_initialized,
        'active_sessions': len(analysis_progress),
        'supported_formats': {
            'image': ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff'],
            'video': ['mp4', 'mov', 'avi', 'mkv', 'webm', 'flv'],
            'audio': ['wav', 'mp3', 'flac', 'aac', 'ogg', 'm4a']
        }
    })

@app.route('/api/initialize', methods=['POST'])
def force_initialize():
    """Force reinitialize models"""
    initialize_models()
    return jsonify({
        'status': 'completed',
        'models_initialized': models_initialized
    })

if __name__ == '__main__':
    # Initialize models before starting the server
    logger.info("Starting Deepfake Detection Service...")
    
    # Create upload directory if it doesn't exist
    from config.settings import Config
    if not os.path.exists(Config.UPLOAD_FOLDER):
        os.makedirs(Config.UPLOAD_FOLDER)
        logger.info(f"Created upload directory: {Config.UPLOAD_FOLDER}")
    
    # Initialize models in a separate thread to avoid blocking startup
    init_thread = threading.Thread(target=initialize_models, daemon=True)
    init_thread.start()
    
    # Start the server
    logger.info("Starting SocketIO server on 0.0.0.0:8000")
    socketio.run(
        app, 
        debug=False,  # Set to False in production
        host='0.0.0.0', 
        port=8000,
        allow_unsafe_werkzeug=True  # For eventlet compatibility
    )