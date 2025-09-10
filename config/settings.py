import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-deepfake-ai'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../temp')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi', 'mkv', 'wav', 'mp3'}
    MODEL_PATH = os.environ.get('MODEL_PATH', './models')
    PYTHON_SERVICE_API_KEY = os.environ.get('PYTHON_SERVICE_API_KEY', '')
    
    # Create upload folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)