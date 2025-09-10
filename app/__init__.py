from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from config.settings import Config

socketio = SocketIO()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Enable CORS
    CORS(app)
    
    # Initialize SocketIO
    socketio.init_app(app, cors_allowed_origins="*", async_mode='eventlet')
    
    # Register blueprints
    from app.routes.analysis import bp as analysis_bp
    from app.routes.health import bp as health_bp
    
    app.register_blueprint(analysis_bp, url_prefix='/api')
    app.register_blueprint(health_bp, url_prefix='/api')
    
    return app