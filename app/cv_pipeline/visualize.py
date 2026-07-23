import cv2
import numpy as np
import base64

def draw_overlay(img: np.ndarray, box_points: np.ndarray, blue_objects: list[dict], classifications: list[str]) -> str:
    """
    Draws the white box outline and the detected blue objects.
    Returns the image as a base64 encoded PNG string.
    """
    # Create a copy so we don't modify the original image directly
    overlay = img.copy()
    
    # Draw white box outline (thick yellow line)
    cv2.polylines(overlay, [box_points], isClosed=True, color=(0, 255, 255), thickness=4)
    
    for i, (obj, classification) in enumerate(zip(blue_objects, classifications)):
        contour = obj["contour"]
        cx, cy = obj["centroid"]
        
        if classification == "inside":
            color = (0, 255, 0) # Green for inside
        elif classification == "outside":
            color = (0, 0, 255) # Red for outside
        else:
            color = (128, 128, 128) # Gray for intersecting (ignored)
            
        # Draw contour
        cv2.drawContours(overlay, [contour], -1, color, 2)
        
        # Draw centroid
        cv2.circle(overlay, (cx, cy), 5, color, -1)
        
        # Draw label
        label = f"#{i+1}"
        cv2.putText(overlay, label, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    # Encode as PNG and then base64
    _, buffer = cv2.imencode('.png', overlay)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    
    return base64_str
