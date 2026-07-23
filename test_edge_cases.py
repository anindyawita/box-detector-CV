import requests
import cv2
import numpy as np

# 1. Non-image file
with open('fake.png', 'w') as f:
    f.write('This is not an image')
r = requests.post('http://localhost:8000/analyze', files={'file': open('fake.png', 'rb')})
print("Non-image:", r.status_code, r.text)

# 2. Image without white box
img = np.ones((500, 500, 3), dtype=np.uint8)*255
cv2.imwrite('no_white_box.png', img)
r = requests.post('http://localhost:8000/analyze', files={'file': open('no_white_box.png', 'rb')})
print("No white box:", r.status_code, r.text)

# 3. Image without blue boxes (but has white box)
img = np.ones((500, 500, 3), dtype=np.uint8)*255
box_rect = ((250, 250), (100, 200), 0)
box = cv2.boxPoints(box_rect)
box = np.int32(box)
cv2.polylines(img, [box], True, (0, 0, 0), 3)
cv2.imwrite('no_blue_box.png', img)
r = requests.post('http://localhost:8000/analyze', files={'file': open('no_blue_box.png', 'rb')})
print("No blue box:", r.status_code, r.text)

# 4. Large image
img = np.ones((4000, 4000, 3), dtype=np.uint8)*255
box_rect = ((2000, 2000), (1000, 2000), 0)
box = cv2.boxPoints(box_rect)
box = np.int32(box)
cv2.polylines(img, [box], True, (0, 0, 0), 3)
cv2.imwrite('large.png', img)
r = requests.post('http://localhost:8000/analyze', files={'file': open('large.png', 'rb')})
print("Large image:", r.status_code, r.text)
