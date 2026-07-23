import cv2
import numpy as np
import os

# Create a blank white image
img = np.ones((500, 500, 3), dtype=np.uint8) * 255

# Draw a thick black rotated rectangle (white box outline)
# We can do this by drawing a polygon
pts = np.array([[100, 100], [400, 150], [350, 450], [50, 400]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=5)

# Draw some blue objects (inside and outside)
# Inside
cv2.rectangle(img, (200, 200), (250, 250), (255, 0, 0), -1) # Blue in BGR
# Outside
cv2.circle(img, (450, 50), 20, (255, 0, 0), -1)

# Ensure the fixtures directory exists
os.makedirs(r"c:\semuacodingan\BlockCounter-CV\energeek-box-detector\tests\fixtures", exist_ok=True)
cv2.imwrite(r"c:\semuacodingan\BlockCounter-CV\energeek-box-detector\tests\fixtures\Gambar1.png", img)
print("Test image generated.")
