import time
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.schemas import AnalyzeResponse
from app.cv_pipeline.preprocess import preprocess_image
from app.cv_pipeline.white_box import detect_white_box
from app.cv_pipeline.blue_box import detect_blue_objects
from app.cv_pipeline.classifier import classify
from app.cv_pipeline.visualize import draw_overlay

app = FastAPI(title="Energeek Box Detector", description="Classical CV Pipeline for detecting objects inside/outside a white box.")

# Mount the static directory for index.html and any sample images if present
import os
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
tests_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Mount tests folder to serve sample images statically for the UI
if os.path.exists(tests_dir):
    app.mount("/tests", StaticFiles(directory=tests_dir), name="tests")

@app.get("/")
async def get_index():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)):
    start_time = time.time()
    
    # Read the uploaded file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=422, detail="Invalid image file.")
        
    try:
        # Run CV Pipeline
        img = preprocess_image(img)
        box_points = detect_white_box(img)
        blue_objects = detect_blue_objects(img)
        inside, outside, classifications = classify(blue_objects, box_points)
        base64_overlay = draw_overlay(img, box_points, blue_objects, classifications)
        
        processing_time_ms = (time.time() - start_time) * 1000.0
        
        return AnalyzeResponse(
            inside_box=inside,
            outside_box=outside,
            total=inside + outside,
            overlay_image_base64=base64_overlay,
            processing_time_ms=processing_time_ms
        )
    except ValueError as e:
        # e.g., White box not detected
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # Catch unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
