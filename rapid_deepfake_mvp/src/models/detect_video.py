# src/models/detect_video.py
import cv2
import logging
from PIL import Image

try:
    from src.models.detect_image_fast import predict_fast
except ImportError:
    pass

logger = logging.getLogger(__name__)

def process_video(video_path: str, num_frames: int = 10) -> dict:
    """Sample frames from video and run image detection. Nodes 6-7."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"label": "invalid", "confidence": 0.0, "consistency_note": "Failed to open video file."}
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return {"label": "invalid", "confidence": 0.0, "consistency_note": "No frames found in video"}
            
        if total_frames < 5:
            return {"label": "invalid", "confidence": 0.0, "consistency_note": "Video too short (less than 5 frames)."}
            
        step = max(1, total_frames // num_frames)
        predictions = []
        
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            try:
                pred = predict_fast(pil_img)
                if pred.get("label") != "unknown":
                    predictions.append(pred)
            except Exception as e:
                logger.error(f"Frame analysis failed: {e}")
                
            if len(predictions) >= num_frames:
                break
                
        cap.release()
        
        if not predictions:
            return {"label": "invalid", "confidence": 0.0, "consistency_note": "Failed to analyze any valid frames."}
        
        fake_conf_sum = sum(p['confidence'] for p in predictions if p['label'] == 'fake')
        real_conf_sum = sum(p['confidence'] for p in predictions if p['label'] == 'real')
        
        fake_count = sum(1 for p in predictions if p['label'] == 'fake')
        real_count = len(predictions) - fake_count
        
        is_fake = fake_count > real_count
        
        if is_fake:
            avg_conf = fake_conf_sum / fake_count if fake_count > 0 else 0.0
        else:
            avg_conf = real_conf_sum / real_count if real_count > 0 else 0.0
            
        return {
            "label": "fake" if is_fake else "real",
            "confidence": round(float(avg_conf), 4),
            "consistency_note": f"{fake_count}/{len(predictions)} frames detected as fake."
        }
    except Exception as e:
        logger.error(f"Video processing error: {e}")
        return {"label": "invalid", "confidence": 0.0, "consistency_note": f"Error processing video: {str(e)}"}
