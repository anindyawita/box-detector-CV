import cv2
import numpy as np

def classify(blue_objects: list[dict], box_points: np.ndarray) -> tuple[int, int, list[str]]:
    """
    Classifies each blue object as 'inside', 'outside', or 'intersecting' the white box.
    Returns (inside_count, outside_count, classifications_list).
    The classifications_list is in the same order as blue_objects.
    """
    inside_count = 0
    outside_count = 0
    classifications = []
    
    for obj in blue_objects:
        c = obj["contour"]
        
        # pointPolygonTest returns:
        # +ve if point is inside
        # 0 if point is on the contour
        # -ve if point is outside
        dists = [cv2.pointPolygonTest(box_points, (float(pt[0][0]), float(pt[0][1])), False) for pt in c]
        
        has_inside = any(d > 0 for d in dists)
        has_outside = any(d < 0 for d in dists)
        
        # If the contour has points strictly inside and strictly outside, it crosses the border
        if has_inside and has_outside:
            classifications.append("intersecting")
        elif has_inside:
            inside_count += 1
            classifications.append("inside")
        elif has_outside:
            outside_count += 1
            classifications.append("outside")
        else:
            # Rare edge case: only has points with d == 0 (exactly on border)
            classifications.append("intersecting")
            
    return inside_count, outside_count, classifications
