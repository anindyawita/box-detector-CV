# Configuration constants for CV Pipeline

# Maximum width to resize the image for consistent processing speed and parameters
MAX_IMAGE_WIDTH = 1024

# Threshold for detecting the white box outline (thick black line).
# Grayscale values below this are considered black.
WHITE_OUTLINE_THRESHOLD = 60

# HSV Range for blue objects
BLUE_HSV_LOWER = (90, 60, 60)
BLUE_HSV_UPPER = (130, 255, 255)

# Kernel size for morphological operations (closing) to merge hatched/striped patterns
MORPH_KERNEL_SIZE = (7, 7)

# Minimum contour area for blue objects to filter out noise
MIN_BLUE_CONTOUR_AREA = 200
