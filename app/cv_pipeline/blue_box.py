import cv2
import numpy as np
from app.config import BLUE_HSV_LOWER, BLUE_HSV_UPPER, MIN_BLUE_CONTOUR_AREA, WHITE_OUTLINE_THRESHOLD

def detect_blue_objects(img: np.ndarray) -> list[dict]:
    """
    Hybrid Pipeline:
    1. Detects objects enclosed by black outlines (good for hatched boxes).
    2. Detects solid blue objects (good for solid boxes without outlines).
    Returns a combined list of dicts containing centroid, contour, and area.
    """
    blue_objects = []
    image_area = img.shape[0] * img.shape[1]
    
    # ALGORITHM 1: Black Outline Segmenter
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, WHITE_OUTLINE_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_outline_contours = []
    if contours and hierarchy is not None:
        for i, c in enumerate(contours):
            # Skip holes inside white regions (these are the outer edges of black lines)
            if hierarchy[0][i][3] != -1:
                continue
                
            area = cv2.contourArea(c)
            if area > 0.9 * image_area:
                continue
            if area < MIN_BLUE_CONTOUR_AREA:
                continue
            valid_outline_contours.append(c)
            
    # Sort by area descending. Index 0 is typically the white box interior.
    valid_outline_contours.sort(key=cv2.contourArea, reverse=True)
    
    candidate_objects = []
    if valid_outline_contours:
        candidate_objects = valid_outline_contours[1:]
        
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, BLUE_HSV_LOWER, BLUE_HSV_UPPER)
    
    # Keep track of areas already detected by Alg 1 to prevent double counting
    alg1_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    for c in candidate_objects:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        
        blue_pixels = cv2.bitwise_and(blue_mask, mask)
        blue_area = cv2.countNonZero(blue_pixels)
        region_area = cv2.contourArea(c)
        
        # If at least 10% of the region contains blue, consider it a blue object
        if region_area > 0 and (blue_area / region_area) > 0.1:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = c[0][0][0], c[0][0][1]
                
            blue_objects.append({
                "centroid": (cx, cy),
                "contour": c,
                "area": float(region_area)
            })
            # Draw this detected object onto our exclusion mask
            cv2.drawContours(alg1_mask, [c], -1, 255, -1)

    # ALGORITHM 2: Solid Color Segmenter
    # Apply a slight morphological close to connect any small gaps in solid boxes
    kernel = np.ones((5, 5), np.uint8)
    blue_mask_closed = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    
    blue_contours, _ = cv2.findContours(blue_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c2 in blue_contours:
        area2 = cv2.contourArea(c2)
        if area2 < MIN_BLUE_CONTOUR_AREA:
            continue
            
        # Check if this contour overlaps with any object already found by Alg 1
        c2_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(c2_mask, [c2], -1, 255, -1)
        
        overlap = cv2.bitwise_and(alg1_mask, c2_mask)
        # If it overlaps at all, it means it's just stripes/parts of an already detected hatched box
        if cv2.countNonZero(overlap) > 0:
            continue 
            
        # If we reach here, it's a new solid blue object!
        M = cv2.moments(c2)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = c2[0][0][0], c2[0][0][1]
            
        blue_objects.append({
            "centroid": (cx, cy),
            "contour": c2,
            "area": float(area2)
        })
        
    return blue_objects
