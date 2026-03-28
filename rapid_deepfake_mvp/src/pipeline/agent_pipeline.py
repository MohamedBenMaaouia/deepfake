# src/pipeline/agent_pipeline.py
from PIL import Image
import os
try:
    from src.models.detect_image_fast import predict_fast
    from src.models.detect_image_strong import predict_strong
    from src.models.detect_video import process_video
    from utils.gradcam_explain import generate_gradcam
except ImportError:
    pass

def run_agentic_pipeline(file_path: str, is_video: bool, intent: str = "quick") -> dict:
    """
    Core agent logic: Branching, early stopping, and escalation.
    """
    result = {}
    
    if is_video:
        # VIDEO PIPELINE
        # DECISION POINT: video processing with frame sampling (Nodes 6-7)
        result = process_video(file_path, num_frames=10)
        
        # Escalation for video (if confidence < 0.85)
        if result['confidence'] < 0.85:
            result['consistency_note'] += " Low confidence, deeper analysis recommended. Agent flagged for escalation."
            
    else:
        # IMAGE PIPELINE
        image = Image.open(file_path).convert("RGB")
        
        # NODE 3: Detect fast with Model A
        res_a = predict_fast(image)
        
        # NODE 4: Confidence Check & Early Stop
        if res_a['confidence'] >= 0.80:
            # DECISION POINT: Early stop (Fast track)
            result = res_a
            result['pipeline'] = 'fast_only_early_stop'
        else:
            # NODE 4 (Escalation): Low confidence -> Model B
            res_b = predict_strong(image)
            gradcam_explanation = generate_gradcam(image)
            
            # NODE 5: Ensemble Check
            if res_a['label'] == res_b['label']:
                # Agree
                avg_conf = (res_a['confidence'] + res_b['confidence']) / 2
                
                # If they agree but it's still low confidence overall -> flag uncertain
                if avg_conf < 0.70:
                    result = {
                        "label": "uncertain / possible real",
                        "confidence": avg_conf,
                        "pipeline": 'ensemble_agree_low_confidence',
                        "uncertainty_flag": True,
                        "explanation": f"Models agreed on {res_a['label']}, but confidence ({avg_conf:.2f}) was too low. " + gradcam_explanation
                    }
                else:
                    result = {
                        "label": res_a['label'],
                        "confidence": min(1.0, avg_conf + 0.05), # increase confidence
                        "pipeline": 'ensemble_agree_escalated',
                        "explanation": gradcam_explanation
                    }
            else:
                # Disagree -> force uncertain
                result = {
                    "label": "uncertain / possible real", 
                    "confidence": (res_a['confidence'] + res_b['confidence']) / 2, 
                    "pipeline": 'ensemble_disagree_escalated',
                    "uncertainty_flag": True,
                    "explanation": f"Models disagreed (A: {res_a['label']}, B: {res_b['label']}). " + gradcam_explanation
                }
                
    # NODE 8: User Intent Adaptation
    output = format_response(result, intent)
    
    # Ethical Warning Node
    if result.get('label') == 'fake' and result.get('confidence', 0) > 0.8:
        output['ethical_warning'] = "WARNING: This media appears to be synthetically generated."
        
    return output

def format_response(result: dict, intent: str) -> dict:
    """Format based on intent: quick, detailed, explanation."""
    label = result.get('label', 'unknown').capitalize()
    conf_pct = int(result.get('confidence', 0) * 100)
    
    if intent == "quick":
        return {"summary": f"{label} ({conf_pct}%)", "raw": result}
        
    elif intent == "detailed":
        details = [
            f"- Prediction: {label}",
            f"- Confidence: {conf_pct}%",
            f"- Pipeline utilized: {result.get('pipeline', 'standard')}"
        ]
        if result.get('uncertainty_flag'):
             details.append("- WARNING: Ensemble disagreement detected")
        if 'consistency_note' in result:
             details.append(f"- Consistency: {result['consistency_note']}")
             
        return {"summary": "\n".join(details), "raw": result}
        
    elif intent == "explanation":
        msg = f"The input was analyzed and classified as {label} with {conf_pct}% confidence."
        if result.get('explanation'):
            msg += f"\n\nAnalysis note: {result['explanation']}"
        elif result.get('pipeline') == 'fast_only_early_stop':
            msg += "\n\nThe fast model was highly confident (>85%), so deep analysis was skipped to save resources."
            
        return {"summary": msg, "raw": result}
            
    return {"summary": f"{label} ({conf_pct}%)", "raw": result}
