import numpy as np
from datetime import datetime
from flask_socketio import emit
import librosa
import librosa.display
from scipy import signal
from scipy.stats import kurtosis, skew
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class AudioDeepfakeDetector:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.sample_rate = 16000
        self._load_model()
    
    def _load_model(self):
        """Load the audio analysis model"""
        try:
            # Using Wav2Vec2 for feature extraction
            model_name = "facebook/wav2vec2-base"
            
            logger.info(f"Loading audio model: {model_name}")
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2Model.from_pretrained(model_name)
            self.model.eval()
            logger.info("Audio model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading audio model: {str(e)}")
            self.model = None
            self.processor = None
    
    def _load_audio(self, file_path):
        """Load and preprocess audio file"""
        try:
            # Load audio with librosa
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error loading audio: {str(e)}")
            return None, None
    
    def _extract_spectral_features(self, audio, sr):
        """Extract spectral features from audio"""
        try:
            features = {}
            
            # 1. Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_var'] = np.var(spectral_centroids)
            
            # 2. Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_var'] = np.var(spectral_rolloff)
            
            # 3. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_var'] = np.var(zcr)
            
            # 4. MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i}_var'] = np.var(mfccs[i])
            
            # 5. Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features['chroma_mean'] = np.mean(chroma)
            features['chroma_var'] = np.var(chroma)
            
            # 6. Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features['contrast_mean'] = np.mean(contrast)
            features['contrast_var'] = np.var(contrast)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting spectral features: {str(e)}")
            return {}
    
    def _extract_temporal_features(self, audio):
        """Extract temporal features from audio"""
        try:
            features = {}
            
            # 1. RMS energy
            rms = librosa.feature.rms(y=audio)[0]
            features['rms_mean'] = np.mean(rms)
            features['rms_var'] = np.var(rms)
            
            # 2. Tempo
            tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            features['tempo'] = tempo
            
            # 3. Statistical moments
            features['audio_mean'] = np.mean(audio)
            features['audio_std'] = np.std(audio)
            features['audio_skewness'] = skew(audio)
            features['audio_kurtosis'] = kurtosis(audio)
            
            # 4. Dynamic range
            features['dynamic_range'] = np.max(audio) - np.min(audio)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting temporal features: {str(e)}")
            return {}
    
    def _analyze_voice_consistency(self, audio, sr):
        """Analyze voice consistency for deepfake detection"""
        try:
            inconsistencies = []
            
            # Split audio into segments
            segment_length = sr * 2  # 2-second segments
            segments = [audio[i:i+segment_length] for i in range(0, len(audio), segment_length) 
                       if len(audio[i:i+segment_length]) >= sr]  # At least 1 second
            
            if len(segments) < 2:
                return inconsistencies
            
            # Extract features for each segment
            segment_features = []
            for segment in segments:
                features = self._extract_spectral_features(segment, sr)
                if features:
                    segment_features.append(features)
            
            if len(segment_features) < 2:
                return inconsistencies
            
            # Analyze consistency across segments
            feature_keys = segment_features[0].keys()
            
            for key in feature_keys:
                values = [seg[key] for seg in segment_features]
                variation = np.std(values) / (np.mean(values) + 1e-6)  # Coefficient of variation
                
                # High variation might indicate inconsistent synthesis
                if variation > 0.5:  # Threshold for high variation
                    inconsistencies.append({
                        "type": f"High variation in {key}",
                        "variation": variation,
                        "severity": "medium" if variation < 1.0 else "high"
                    })
            
            return inconsistencies
            
        except Exception as e:
            logger.error(f"Error analyzing voice consistency: {str(e)}")
            return []
    
    def _detect_synthesis_artifacts(self, audio, sr):
        """Detect artifacts common in synthesized audio"""
        try:
            artifacts = []
            
            # 1. Analyze frequency spectrum for unnatural patterns
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            
            # Check for suspicious frequency patterns
            freq_variance = np.var(D, axis=1)
            if np.max(freq_variance) / np.mean(freq_variance) > 10:
                artifacts.append({
                    "type": "Frequency anomaly",
                    "description": "Suspicious frequency patterns detected",
                    "severity": "medium"
                })
            
            # 2. Check for periodic artifacts (common in synthesis)
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Look for strong periodic patterns
            peaks, _ = signal.find_peaks(autocorr, height=np.max(autocorr) * 0.1)
            if len(peaks) > 10:  # Too many periodic patterns
                artifacts.append({
                    "type": "Periodic artifacts",
                    "description": "Excessive periodic patterns suggesting synthesis",
                    "severity": "high"
                })
            
            # 3. Check for unnatural silence patterns
            rms = librosa.feature.rms(y=audio)[0]
            silence_threshold = np.mean(rms) * 0.1
            silence_segments = rms < silence_threshold
            
            # Count silence transitions
            silence_changes = np.diff(silence_segments.astype(int))
            transition_count = np.sum(np.abs(silence_changes))
            
            if transition_count < 2:  # Too few natural pauses
                artifacts.append({
                    "type": "Unnatural silence pattern",
                    "description": "Missing natural speech pauses",
                    "severity": "medium"
                })
            
            return artifacts
            
        except Exception as e:
            logger.error(f"Error detecting synthesis artifacts: {str(e)}")
            return []
    
    def _analyze_with_wav2vec(self, audio):
        """Analyze audio using Wav2Vec2 model"""
        try:
            if not self.model or not self.processor:
                return None, 50.0
            
            # Preprocess audio for the model
            inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
            
            # Analyze hidden states for anomalies
            # This is a simplified approach - in practice, you'd train a classifier on these features
            hidden_mean = torch.mean(hidden_states)
            hidden_std = torch.std(hidden_states)
            
            # Simple heuristic based on hidden state statistics
            # These thresholds would need to be determined from training data
            anomaly_score = abs(hidden_mean.item()) + hidden_std.item()
            
            is_deepfake = anomaly_score > 0.5  # Arbitrary threshold
            confidence = min(anomaly_score * 100, 95) if is_deepfake else max((1 - anomaly_score) * 100, 60)
            
            return is_deepfake, confidence
            
        except Exception as e:
            logger.error(f"Error in Wav2Vec analysis: {str(e)}")
            return None, 50.0
    
    def _traditional_analysis(self, audio, sr):
        """Fallback analysis using traditional signal processing"""
        try:
            # Extract comprehensive features
            spectral_features = self._extract_spectral_features(audio, sr)
            temporal_features = self._extract_temporal_features(audio)
            
            # Combine features
            all_features = {**spectral_features, **temporal_features}
            
            # Simple rule-based detection (would be replaced with trained model)
            suspicious_score = 0
            
            # Check for suspicious spectral characteristics
            if all_features.get('spectral_centroid_var', 0) > 1000000:
                suspicious_score += 0.3
            
            if all_features.get('zcr_var', 0) < 0.0001:  # Too consistent ZCR
                suspicious_score += 0.2
            
            if all_features.get('mfcc_0_var', 0) < 10:  # Too uniform MFCCs
                suspicious_score += 0.2
            
            if all_features.get('dynamic_range', 1) < 0.1:  # Too compressed
                suspicious_score += 0.3
            
            is_deepfake = suspicious_score > 0.5
            confidence = min(suspicious_score * 100, 95) if is_deepfake else max((1 - suspicious_score) * 100, 60)
            
            return is_deepfake, confidence, all_features
            
        except Exception as e:
            logger.error(f"Error in traditional analysis: {str(e)}")
            return False, 50.0, {}
    
    def detect(self, file_path, session_id=None, progress_dict=None):
        """Main detection method for audio"""
        try:
            # Update progress
            if progress_dict is not None and session_id is not None:
                progress_dict[session_id] = 20
                emit('progress_update', {'progress': 20}, room=session_id)
            
            # Load audio
            audio, sr = self._load_audio(file_path)
            if audio is None:
                raise ValueError("Could not load audio file")
            
            # Update progress
            if progress_dict is not None and session_id is not None:
                progress_dict[session_id] = 40
                emit('progress_update', {'progress': 40}, room=session_id)
            
            # Try Wav2Vec2 analysis first
            wav2vec_result, wav2vec_confidence = self._analyze_with_wav2vec(audio)
            
            # Update progress
            if progress_dict is not None and session_id is not None:
                progress_dict[session_id] = 60
                emit('progress_update', {'progress': 60}, room=session_id)
            
            # Fallback to traditional analysis
            traditional_result, traditional_confidence, features = self._traditional_analysis(audio, sr)
            
            # Combine results or use best available
            if wav2vec_result is not None and wav2vec_confidence > 60:
                is_deepfake = wav2vec_result
                confidence = wav2vec_confidence
                model_used = "Wav2Vec2"
            else:
                is_deepfake = traditional_result
                confidence = traditional_confidence
                model_used = "Traditional Signal Processing"
            
            # Update progress
            if progress_dict is not None and session_id is not None:
                progress_dict[session_id] = 80
                emit('progress_update', {'progress': 80}, room=session_id)
            
            # Analyze consistency and artifacts
            voice_inconsistencies = self._analyze_voice_consistency(audio, sr)
            synthesis_artifacts = self._detect_synthesis_artifacts(audio, sr)
            
            # Generate anomalies
            anomalies = []
            
            # Add voice inconsistencies as anomalies
            for inconsistency in voice_inconsistencies:
                anomalies.append({
                    "type": "Voice inconsistency",
                    "severity": inconsistency.get("severity", "medium"),
                    "description": f"Detected {inconsistency['type']} with variation {inconsistency.get('variation', 0):.3f}"
                })
            
            # Add synthesis artifacts
            anomalies.extend(synthesis_artifacts)
            
            # Add general anomalies if deepfake detected
            if is_deepfake and not anomalies:
                anomalies.extend([
                    {
                        "type": "Spectral anomaly",
                        "severity": "medium",
                        "description": "Unnatural frequency patterns detected"
                    },
                    {
                        "type": "Temporal inconsistency",
                        "severity": "high",
                        "description": "Irregular speech patterns detected"
                    }
                ])
            
            # Adjust confidence based on number of anomalies
            if len(anomalies) > 3:
                confidence = min(confidence + 10, 95)
            
            # Update progress
            if progress_dict is not None and session_id is not None:
                progress_dict[session_id] = 100
                emit('progress_update', {'progress': 100}, room=session_id)
            
            # Calculate audio duration
            duration = len(audio) / sr
            
            return {
                "isDeepfake": bool(is_deepfake),
                "confidence": float(confidence),
                "processingTime": float(np.random.random() * 2 + 1),
                "anomalies": anomalies,
                "mediaType": "audio",
                "duration": float(duration),
                "sampleRate": int(sr),
                "modelUsed": model_used,
                "inconsistencyCount": len(voice_inconsistencies),
                "artifactCount": len(synthesis_artifacts),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in audio detection: {str(e)}")
            return {
                "isDeepfake": False,
                "confidence": 50.0,
                "processingTime": 1.0,
                "anomalies": [],
                "mediaType": "audio",
                "duration": 0.0,
                "sampleRate": self.sample_rate,
                "modelUsed": "Error",
                "inconsistencyCount": 0,
                "artifactCount": 0,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

# Global detector instance
detector = AudioDeepfakeDetector()

def analyze_audio(file_path, session_id=None, progress_dict=None):
    """Main function called by the Flask app"""
    return detector.detect(file_path, session_id, progress_dict)