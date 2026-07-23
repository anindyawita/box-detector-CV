import cv2
import numpy as np
from app.config import MAX_IMAGE_WIDTH

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Resizes image to a max width while maintaining aspect ratio,
    and applies a slight blur to reduce noise.
    """
    h, w = img.shape[:2]
    if w > MAX_IMAGE_WIDTH:
        scale = MAX_IMAGE_WIDTH / w
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Optional light blur to reduce noise before thresholding
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img
