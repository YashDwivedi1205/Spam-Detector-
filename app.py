import os
import threading
import time

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.pipeline.prediction_pipeline import PredictionPipeline
app = FastAPI(title="Spam Detector")
#CSS and later images are served from the static folder
app.mount("/static", StaticFiles(directory="static"), name= "static")

#HTML pages comes from the templates/folder
templates = Jinja2Templates(directory="templates")

#Model is loaded once, when the app starts not on every request
predictor = PredictionPipeline()

 #Training is OFF unless this environment variable is set to 1.
# On a hosted server leave it unset: anyone could otherwise start training.
TRAINING_ENABLED = os.getenv("ENABLE_TRAINING", "0") == "1"

training = {
    "status": "idle",        # idle, running, done or failed
    "started_at": None,
    "finished_at": None,
    "metrics": None,
    "model_path": None,
    "error": None,
}
training_lock = threading.Lock()
 
def render_page(request: Request, **context):
    context["training_enabled"] = TRAINING_ENABLED
    return templates.TemplateResponse(request, "index.html", context)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Show the empty form"""
    return render_page(request)

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, text: str = Form("")):
    """Check one message and show the verdict on the same page."""
    try:
        result = predictor.predict_one(text)
        return render_page(request, text=text, result=result)

    except ValueError as e:
        # Empty message: a user mistake, so show the message instead of crashing
        return render_page(request, text=text, error = str(e))

@app.get("/health")
def health():
    """Used later by the hosting service to check the app is running."""
    return {"status": "ok"}
    
#Training
def run_training():
    """Runs in a background thread, so the app keeps answering requests."""
    try:
        # Imported here so a prediction-only server never needs pymongo or pandas
        from src.pipeline.training_pipeline import TrainPipeline
 
        artifact = TrainPipeline().run_pipeline()
 
        metrics = None
        model_path = None
        if artifact is not None:
            metrics = {
                "train_accuracy": artifact.train_accuracy,
                "test_accuracy": artifact.test_accuracy,
                "test_precision": artifact.test_precision,
                "test_recall": artifact.test_recall,
            }
            model_path = artifact.trained_model_file_path
 
        with training_lock:
            training.update(status="done", finished_at=time.time(),
                            metrics=metrics, model_path=model_path)
    except Exception as e:
        with training_lock:
            training.update(status="failed", finished_at=time.time(), error=str(e))
 
 
@app.get("/train/status")
def train_status():
    """Current state of training, polled by the Train model tab."""
    with training_lock:
        snapshot = dict(training)
 
    elapsed = None
    if snapshot["started_at"] is not None:
        end = snapshot["finished_at"] or time.time()
        elapsed = round(end - snapshot["started_at"])
 
    snapshot["enabled"] = TRAINING_ENABLED
    snapshot["elapsed_seconds"] = elapsed
    return snapshot
 
 
@app.post("/train/start")
def train_start():
    """Start training in the background (only when ENABLE_TRAINING=1)."""
    if not TRAINING_ENABLED:
        return JSONResponse({"detail": "Training is turned off on this server."}, status_code=403)
 
    with training_lock:
        if training["status"] == "running":
            return JSONResponse({"detail": "Training is already running."}, status_code=409)
        training.update(status="running", started_at=time.time(), finished_at=None,
                        metrics=None, model_path=None, error=None)
 
    threading.Thread(target=run_training, daemon=True).start()
    return {"status": "running"}  