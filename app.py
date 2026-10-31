from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Show the empty form"""
    return templates.TemplateResponse(request, "index.html", {})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, text: str = Form("")):
    """Check one message and show teh verdict on the same page."""
    try:
        result = predictor.predict_one(text)
        context = {"text":text, "result": result}

    except ValueError as e:
        context = {"text": text, "error": str(e)}
    return templates.TemplateResponse(request, "index.html", context)

@app.get("/health")
def health():
    """Used later by the hosting service to check the app is running."""
    return {"status": "ok"}
    
    