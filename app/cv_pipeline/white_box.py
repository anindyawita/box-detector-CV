import cv2
import numpy as np
from app.config import WHITE_OUTLINE_THRESHOLD

def detect_white_box(img: np.ndarray) -> np.ndarray:
    """
    Detects the white box by finding the largest non-background region bounded by black outlines.
    Returns the 4 corner points of the rotated bounding box.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use THRESH_BINARY: black outlines become 0, all regions (background, box interiors) become 255
    _, thresh = cv2.threshold(gray, WHITE_OUTLINE_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours or hierarchy is None:
        raise ValueError("White box not detected. Please check image contrast/lighting.")
        
    image_area = img.shape[0] * img.shape[1]
    valid_contours = []
    
    for i, c in enumerate(contours):
        # Skip holes inside white regions (these are the outer edges of black lines)
        if hierarchy[0][i][3] != -1:
            continue
            
        area = cv2.contourArea(c)
        # Skip the massive background contour
        if area > 0.9 * image_area:
            continue
        # Skip small noise
        if area < 1000:
            continue
        valid_contours.append(c)
        
    if not valid_contours:
        raise ValueError("White box not detected. Please check image contrast/lighting.")
        
    # The largest valid region bounded by black lines should be the white box interior
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # Use minAreaRect to handle rotated boxes
    rect = cv2.minAreaRect(largest_contour)
    
    # Validate the rect
    if rect[1][0] < 10 or rect[1][1] < 10:
        raise ValueError("White box not detected properly (invalid size).")
        
    box_points = cv2.boxPoints(rect)
    box_points = np.int32(box_points)
    
    return box_points
