import cv2
import numpy as np
from flask_socketio import emit
from datetime import datetime

def analyze_video(file_path, session_id=None, progress_dict=None):
    """Analyze video for deepfake detection"""
    # Open video file
    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_analysis = []
    anomalies = []
    
    # Process each frame
    for frame_num in range(min(100, total_frames)):  # Limit to first 100 frames for demo
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update progress
        if progress_dict is not None and session_id is not None:
            progress = 10 + (frame_num / min(100, total_frames)) * 80
            progress_dict[session_id] = progress
            emit('progress_update', {'progress': progress}, room=session_id)
        
        # Simulate frame analysis (replace with actual model)
        is_deepfake_frame = np.random.random() > 0.6
        confidence = np.random.random() * 20 + (80 if is_deepfake_frame else 75)
        
        frame_analysis.append({
            "frame": frame_num,
            "confidence": float(confidence),
            "isDeepfake": bool(is_deepfake_frame)
        })
        
        # Add anomaly if detected
        if is_deepfake_frame and np.random.random() > 0.7:
            anomalies.append({
                "type": "Temporal inconsistency",
                "severity": "high",
                "description": f"Inconsistent facial movements at frame {frame_num}",
                "frame": frame_num
            })
    
    cap.release()
    
    # Determine overall result
    deepfake_frames = sum(1 for f in frame_analysis if f['isDeepfake'])
    is_deepfake = deepfake_frames / len(frame_analysis) > 0.3
    overall_confidence = np.mean([f['confidence'] for f in frame_analysis])
    
    # Update progress
    if progress_dict is not None and session_id is not None:
        progress_dict[session_id] = 95
        emit('progress_update', {'progress': 95}, room=session_id)
    
    # Return result
    return {
        "isDeepfake": bool(is_deepfake),
        "confidence": float(overall_confidence),
        "processingTime": float(np.random.random() * 3 + 2),
        "anomalies": anomalies,
        "frameAnalysis": frame_analysis,
        "totalFrames": total_frames,
        "fps": fps,
        "mediaType": "video",
        "timestamp": datetime.now().isoformat()
    }