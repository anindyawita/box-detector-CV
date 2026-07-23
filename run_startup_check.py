import os
import cv2
import sys

# Add the project root to sys.path so we can import app modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.cv_pipeline.preprocess import preprocess_image
from app.cv_pipeline.white_box import detect_white_box
from app.cv_pipeline.blue_box import detect_blue_objects
from app.cv_pipeline.classifier import classify

def run_check():
    print("=== Starting Regression Check ===")
    fixtures_dir = os.path.join(os.path.dirname(__file__), "tests", "fixtures")
    
    if not os.path.exists(fixtures_dir):
        print(f"WARNING: Fixtures directory not found at {fixtures_dir}")
        return
        
    for i in range(1, 5):
        filename = f"Gambar{i}.png"
        filepath = os.path.join(fixtures_dir, filename)
        
        # fallback to jpg if png not found
        if not os.path.exists(filepath):
            filename = f"Gambar{i}.jpg"
            filepath = os.path.join(fixtures_dir, filename)
            
        if not os.path.exists(filepath):
            print(f"WARNING: Test image {filename} not found in {fixtures_dir}. Skipping.")
            continue
            
        try:
            img = cv2.imread(filepath)
            if img is None:
                print(f"WARNING: Could not read image {filename}.")
                continue
                
            img = preprocess_image(img)
            box_points = detect_white_box(img)
            blue_objects = detect_blue_objects(img)
            inside, outside, _ = classify(blue_objects, box_points)
            
            print(f"[OK] {filename} -> Inside: {inside}, Outside: {outside}, Total: {inside + outside}")
        except Exception as e:
            print(f"[ERROR] Failed processing {filename}: {str(e)}")
            
    print("=== Regression Check Finished ===")

if __name__ == "__main__":
    run_check()
