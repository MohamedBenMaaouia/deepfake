# src/models/detect_image_fast.py
import logging
from transformers import pipeline
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    pipe = pipeline("image-classification", model="prithivMLmods/deepfake-detector-model-v1")
    logger.info("Successfully loaded fast model (Model A).")
except Exception as e:
    logger.error(f"Failed to load fast model: {e}")
    pipe = None

def predict_fast(image: Image.Image) -> dict:
    """Run fast detection on a single image. Node 3."""
    if pipe is None:
        logger.warning("Fast model pipeline is None. Returning dummy data.")
        return {"label": "fake", "confidence": 0.95}

    try:
        results = pipe(image)
        logger.info(f"[Model A] Raw output: {results}")
        best_result = results[0]
        label_raw = best_result['label'].lower()
        score = float(best_result['score'])
        # Default handling: find score for 'fake' explicitly if present, else use logic
        fake_score = 0.0
        real_score = 0.0
        
        for res in results:
            lbl = res['label'].lower()
            if 'fake' in lbl or 'spoof' in lbl:
                fake_score = max(fake_score, float(res['score']))
            elif 'real' in lbl or 'live' in lbl or 'original' in lbl:
                real_score = max(real_score, float(res['score']))
                
        # If neither is found, fallback to the top prediction's mapping
        if fake_score == 0.0 and real_score == 0.0:
            if 'fake' in label_raw or 'spoof' in label_raw:
                fake_score = score
            else:
                real_score = score
                
        # Make a decision based on fake probability
        # Invert probability if real is super high and fake is low
        if fake_score > real_score:
            label = 'fake'
            confidence = fake_score
        else:
            label = 'real'
            confidence = real_score
            
        return {"label": label, "confidence": confidence}
    except Exception as e:
        logger.error(f"Error during fast prediction: {e}")
        return {"label": "unknown", "confidence": 0.0}
