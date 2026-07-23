import os
import sys
import cv2
import pytest
import numpy as np

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cv_pipeline.preprocess import preprocess_image
from app.cv_pipeline.white_box import detect_white_box
from app.cv_pipeline.blue_box import detect_blue_objects
from app.cv_pipeline.classifier import classify

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

def get_fixture_path(idx):
    path_png = os.path.join(FIXTURES_DIR, f"Gambar{idx}.png")
    path_jpg = os.path.join(FIXTURES_DIR, f"Gambar{idx}.jpg")
    if os.path.exists(path_png): return path_png
    if os.path.exists(path_jpg): return path_jpg
    return None

@pytest.mark.parametrize("img_idx", [1, 2, 3, 4])
def test_end_to_end_pipeline(img_idx):
    img_path = get_fixture_path(img_idx)
    if not img_path:
        pytest.skip(f"Fixture image Gambar{img_idx} not found. Please add to tests/fixtures/")
        
    img = cv2.imread(img_path)
    assert img is not None, f"Failed to read {img_path}"
    
    # 1. Preprocess
    img = preprocess_image(img)
    assert img.shape[1] <= 1024, "Image width should be <= 1024 after preprocessing"
    
    # 2. White Box Detection
    box_points = detect_white_box(img)
    assert box_points is not None
    assert box_points.shape == (4, 2), "Should return 4 corner points (rotated rect)"
    
    # 3. Blue Object Detection
    blue_objects = detect_blue_objects(img)
    assert isinstance(blue_objects, list)
    
    # 4. Classification
    inside, outside, classifications = classify(blue_objects, box_points)
    
    assert inside >= 0
    assert outside >= 0
    assert inside + outside <= len(blue_objects)
    assert len(classifications) == len(blue_objects)
