# src/api/main.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from src.pipeline.agent_pipeline import run_agentic_pipeline

app = FastAPI(title="Deepfake MVP Agent API")

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze_media(
    file: UploadFile = File(...),
    intent: str = Form("quick")
):
    """
    Endpoint (Node 1 Gate) to receive image/video and trigger the agentic pipeline.
    """
    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    content_type = file.content_type
    
    # Node 1 Classification
    if "image" in content_type:
        is_video = False
    elif "video" in content_type:
        is_video = True
    else:
        os.remove(file_path)
        return {"error": "Invalid file type. Please upload Image or Video."}
        
    # Run Agentic Logic
    try:
        response = run_agentic_pipeline(file_path, is_video=is_video, intent=intent)
    except Exception as e:
        response = {"error": str(e)}
    finally:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
            
    return response

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Deepfake MVP API is running."}
