Viewed SKILL.md:1-392

# ⚡ Rapid Core Deepfake Detector (Agentic MVP)

## 📌 1. Overview & Design Goal
This project is a 2-week **Minimum Viable Product (MVP)** for synthetic media (Deepfake) detection. It is designed as an **agentic system**, meaning it does not rigidly process every file from top to bottom. Instead, the agent makes intelligent decisions based on media modality (Image vs. Video) and prediction confidence. 

The core design principle is **Confidence-based Escalation**: inexpensive, fast checks are used first, and more expensive, heavier analysis (including Explainable AI) is triggered only when the fast checks yield low confidence.

**Key Features:**
- **Hybrid Pipeline:** Modality routing (Image/Video).
- **Ensemble Decision Making:** Model A (fast) and Model B (strong) cross-validation.
- **Agentic Routing:** Early-stopping mechanism to save compute.
- **Explainability:** Grad-CAM heatmaps for escalated image analysis.

---

## 🏗️ 2. Architecture Flow Diagram

```text
[User Uploads Image/Video]
           │
           ▼
    +--------------+
    | Streamlit UI | (Frontend)
    +--------------+
           │
           ▼ (HTTP POST)
    +--------------+
    | FastAPI Gate | (Node 1: Input Val & Modality Check)
    +--------------+
           │
           ├──► [Video] ──► detect_video.py (Extract Frames ──► predict_fast)
           │
           └──► [Image] ──► agent_pipeline.py
                                   │
                                   ▼
                            predict_fast() [Model A]
                                   │
                               [Confidence > 0.80?]
                                   ├──► [YES] ──► Fast Track Return (Early Stop)
                                   │
                                   └──► [NO]  ──► predict_strong() [Model B] + Grad-CAM
                                                         │
                                                  [Ensemble Check]
                                                         ├──► Agree (>0.70 avg) --> Output Label
                                                         └──► Disagree/Low --> "Uncertain / Possible Real"
                                                         
                                                         
           ┌─────────────────────────────────────────────┐
           ▼                                             ▼
  [Detailed Response JSON]                      [Grad-CAM Heatmap]
           │
           ▼
    +--------------+
    | Streamlit UI | (Displays final classification, confidence, and internal logs)
    +--------------+
```

---

## 📂 3. File-by-File Breakdown

### Infrastructure
- **[Dockerfile](cci:7://file:///C:/Users/medbe/OneDrive/Documents/projet%20federateur%202/rapid_deepfake_mvp/Dockerfile:0:0-0:0)**: Container definition for deployment. Runs both the FastAPI backend (`uvicorn`) and the Streamlit frontend concurrently using `bash -c`.
- **[requirements.txt](cci:7://file:///C:/Users/medbe/OneDrive/Documents/projet%20federateur%202/rapid_deepfake_mvp/requirements.txt:0:0-0:0)**: Project dependencies including `transformers`, `torch`, `opencv-python`, `fastapi`, `streamlit`, `pytorch-grad-cam`, and `timm`.

### Backend API
- **[src/api/main.py](cci:7://file:///C:/Users/medbe/OneDrive/Documents/projet%20federateur%202/rapid_deepfake_mvp/src/api/main.py:0:0-0:0)**: The main FastAPI entry point (`Node 1`). Contains the `/analyze` endpoint which securely saves the uploaded file to `temp_uploads`, determines whether the file is an image or video based on the MIME type, and triggers the `agent_pipeline`. Cleans up the file from the disk after execution.

### Agent Logic
- **[src/pipeline/agent_pipeline.py](cci:7://file:///C:/Users/medbe/OneDrive/Documents/projet%20federateur%202/rapid_deepfake_mvp/src/pipeline/agent_pipeline.py:0:0-0:0)**: The "brain" of the project. Implements the decision tree. It imports the model handlers, routes traffic, manages the early stopping threshold (`0.80`), requests Grad-CAM for low-confidence images, evaluates ensemble agreement, overrides outputs for uncertainty, and formats the response based on user intent (`quick`, `detailed`, `explanation`). Includes a hardcoded ethical warning node for highly confident deepfake detections.

### Machine Learning Models
- **[src/models/detect_image_fast.py](cci:7://file:///C:/Users/medbe/OneDrive/Documents/projet%20federateur%202/rapid_deepfake_mvp/src/models/detect_image_fast.py:0:0-0:0)**: Node 3 handler utilizing Hugging Face's [pipeline](cci:1://file:///C:/Users/medbe/OneDrive/Documents/projet%20federateur%202/rapid_deepfake_mvp/src/pipeline/agent_pipeline.py:11:0-81:17). Loads Model A (`prithivMLmods/deepfake-detector-model-v1`). Parses raw float scores rigorously, searching specifically for `fake` vs `real` strings internally. Serves as the first line of defense.
- **[src/models/detect_image_strong.py](cci:7://file:///C:/Users/medbe/OneDrive/Documents/projet%20federateur%202/rapid_deepfake_mvp/src/models/detect_image_strong.py:0:0-0:0)**: Node 4 Escalation handler. Loads Model B (`dima806/deepfake_vs_real_image_detection`). It is only invoked if Model A lacks confidence.
- **[src/models/detect_video.py](cci:7://file:///C:/Users/medbe/OneDrive/Documents/projet%20federateur%202/rapid_deepfake_mvp/src/models/detect_video.py:0:0-0:0)**: Video processing handler. Secures a frame count and safely extracts equidistant frames (`default 10`, `minimum 5`). Passes frames iteratively to [predict_fast](cci:1://file:///C:/Users/medbe/OneDrive/Documents/projet%20federateur%202/rapid_deepfake_mvp/src/models/detect_image_fast.py:15:0-57:54), aggregates total detections, and averages probabilities using voting logic to produce a single aggregated prediction for the video.

### Explainability
- **[utils/gradcam_explain.py](cci:7://file:///C:/Users/medbe/OneDrive/Documents/projet%20federateur%202/rapid_deepfake_mvp/utils/gradcam_explain.py:0:0-0:0)**: Implements Explainable AI for Model B by generating a heatmap indicating which regions the CNN/ViT attended to for its prediction. Features a defensive [HuggingFaceModelWrapper](cci:2://file:///C:/Users/medbe/OneDrive/Documents/projet%20federateur%202/rapid_deepfake_mvp/utils/gradcam_explain.py:31:8-37:43) to bypass Hugging Face `ImageClassifierOutput` errors, maps tensors to the correct `device` safely, and saves visual outputs to local storage.

### Frontend
- **[streamlit_app/app.py](cci:7://file:///C:/Users/medbe/OneDrive/Documents/projet%20federateur%202/rapid_deepfake_mvp/streamlit_app/app.py:0:0-0:0)**: The interactive visualization dashboard. Allows users to upload media constraint to MP4, JPG, or PNG, select an output verbosity (`Intent`), render the resulting AI summary, display ethical warnings, and inspect the raw JSON trace from the agent explicitly.

---

## 📁 4. Runtime Folders

- **`temp_uploads/`**: Created automatically at runtime by [main.py](cci:7://file:///C:/Users/medbe/OneDrive/Documents/projet%20federateur%202/rapid_deepfake_mvp/src/api/main.py:0:0-0:0) and [gradcam_explain.py](cci:7://file:///C:/Users/medbe/OneDrive/Documents/projet%20federateur%202/rapid_deepfake_mvp/utils/gradcam_explain.py:0:0-0:0). 
   - Stores raw MP4/JPG incoming files while the agent analyzes them (these are wiped via `os.remove` natively).
   - Stores the generated Grad-CAM heatmap visualization states (`gradcam_<uuid>.jpg`). These are *not* automatically wiped in the current iteration to allow manual inspection / UI rendering.

---

## 🌳 5. Agentic Decision Tree Logic

```text
1. Modality Gate:
   - If Video -> Jump to video processing.
   - If Image -> Proceed to Model A.

2. Initial State (Image):
   - A = predict_fast(Image)

3. Confidence Escalation (Threshold = 0.80):
   - IF A.confidence >= 0.80:
         RETURN A (Early Stop activated)
   - ELSE:
         B = predict_strong(Image)
         Heatmap = generate_gradcam(Image)

4. Ensemble Check (Models A and B):
   - IF A.label == B.label:
         - IF avg(A.confidence, B.confidence) >= 0.70:
               RETURN Agreed Label with slightly boosted confidence.
         - ELSE:
               RETURN "uncertain / possible real" (Confidence too low despite agreement)
   - ELSE (Disagreement):
         RETURN "uncertain / possible real"
```

---

## ⚠️ 6. Known Issues

1. **ViT Grad-CAM Target Layer Fragility:** While `target_layers` fallback logic exists, heavily nested Vision Transformers (ViTs) update their architecture occasionally in the Hugging Face `transformers` backend. If `pytorch-grad-cam` misses the final classification token dimension, heatmaps may fail to generate visually (falls back to a text summary safely).
2. **Classification Bias on Artifacts:** Depending on the training distribution of `prithivMLmods` and `dima806`, highly compressed JPEGs or low-quality *real* photos can still sometimes be penalized by the model due to heavy compression artifacts simulating generative noise.
3. **Heatmap Cleanup:** `temp_uploads` currently accumulates `.jpg` heatmaps indefinitely over time.
4. **Blocking Inference:** The OpenCV frame extraction and Hugging Face pipelines operate synchronously inside FastAPI standard routes. Under heavy parallel load, this blocks the event loop.

---

## 🚀 7. How to Run & Test

**Terminal 1: Start Backend API**
```bash
uvicorn src.api.main:app --reload
```
*(Runs on `http://127.0.0.1:8000`. You can test API health at `/`)*

**Terminal 2: Start Frontend UI**
```bash
streamlit run streamlit_app/app.py
```
*(Runs on `http://localhost:8501`. Opens automatically in browser.)*

### Expected Test Cases:
1. **High Definition Real Face (Image):** Should immediately route to `fast_only_early_stop` yielding a "Real" label. No heatmap generated.
2. **AI Generated Image (Midjourney/Stable Diffusion):** Model A flags it > 0.80 fake. Returns early with high score.
3. **Ambiguous/Low-Res Real Image:** Model A scores `0.65 Real`. Triggers escalation. Model B scores `0.60 Real`. They agree, but average confidence (`0.625`) < `0.70`, triggering `"uncertain / possible real"` flag with an underlying heatmap.
4. **Video:** Upload 5-second `mp4`. Should sample up to 10 frames cleanly. Returns combined aggregate score. Upload missing/broken MP4 -> returns validation error "Video too short".

---

## 🔮 8. Future Improvements

1. **Asynchronous Inference & Job Queues:** Wrap model executions in `asyncio.to_thread` or utilize `Celery/Redis`. Deepfake pipelines can take 3–5 seconds and block normal routing concurrency.
2. **Dynamic Grad-CAM Parsing:** Implement robust dictionary/key tracing for ViT pooling layers to permanently stabilize `target_layers` selection on arbitrary model strings.
3. **Audio Discrepancy Checks:** Extend [detect_video.py](cci:7://file:///C:/Users/medbe/OneDrive/Documents/projet%20federateur%202/rapid_deepfake_mvp/src/models/detect_video.py:0:0-0:0) to strip the `.wav` audio layer from the MP4, run a lightweight synthetic audio classifier (like `Wav2Vec2`), and factor the audio-visual sync into the ensemble confidence score.
4. **Automated Heatmap Pruning:** Implement a background Cron worker or `FastAPI BackgroundTasks` to purge image overlays older than 1 hour in `/temp_uploads` to prevent memory/storage leakage.
5. **Model Fine-Tuning Calibration:** Gather local datasets of failure cases (where real images get flagged) and implement post-processing temperature scaling calibrated around a `0.0` to `1.0` logistic regression bound for fairer mapping instead of raw `score` usage.
