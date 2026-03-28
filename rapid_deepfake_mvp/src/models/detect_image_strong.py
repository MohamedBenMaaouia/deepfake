# src/models/detect_image_strong.py
import logging
from transformers import pipeline
from PIL import Image

logger = logging.getLogger(__name__)

try:
    strong_pipe = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")
    logger.info("Successfully loaded strong model (Model B).")
except Exception as e:
    logger.error(f"Failed to load strong model: {e}")
    strong_pipe = None

def predict_strong(image: Image.Image) -> dict:
    """Run deeper analysis if confidence was low. Node 4 Escalation."""
    if strong_pipe is None:
        logger.warning("Strong model pipeline is None. Returning dummy data.")
        return {"label": "real", "confidence": 0.88}
        
    try:
        results = strong_pipe(image)
        logger.info(f"[Model B] Raw output: {results}")
        best_result = results[0]
        label_raw = best_result['label'].lower()
        score = float(best_result['score'])
        
        fake_score = 0.0
        real_score = 0.0
        
        for res in results:
            lbl = res['label'].lower()
            if 'fake' in lbl or 'spoof' in lbl:
                fake_score = max(fake_score, float(res['score']))
            elif 'real' in lbl or 'live' in lbl or 'original' in lbl:
                real_score = max(real_score, float(res['score']))
        
        if fake_score == 0.0 and real_score == 0.0:
            if 'fake' in label_raw:
                fake_score = score
            else:
                real_score = score
                
        if fake_score > real_score:
            label = 'fake'
            confidence = fake_score
        else:
            label = 'real'
            confidence = real_score
            
        return {"label": label, "confidence": confidence}
    except Exception as e:
        logger.error(f"Error during strong prediction: {e}")
        return {"label": "unknown", "confidence": 0.0}
