from src.cameracalibration import CalibrateCamera
from src.imagecalibration import PixelWorldMapper
import numpy as np
import cv2

img_mask = "./images/*.jpeg"
image = cv2.imread('./images/checkerboard.jpeg')
known_grid_size = 3
pattern_size = (4,4)
figsize = (20, 20)
calibrator = CalibrateCamera(img_mask,figsize, known_grid_size, pattern_size)
rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = calibrator.calibrate()

print(f"RMS: {rms}")
rvecs = np.array(_rvecs, dtype=np.float32).reshape(3, 1)
tvecs = np.array(_tvecs, dtype=np.float32).reshape(3, 1)

mapper = PixelWorldMapper(camera_matrix, dist_coeffs, known_grid_size, image, pattern_size, rvecs, tvecs)

point1 = (400, 190)
point2 = (400, 270)
distance = mapper.distance_between_pixels(point1, point2)
print(f"Distance entre les points: {distance *100} cm")
cv2.circle(image, point1, radius=5, color=(0, 0, 255), thickness=-1)  # Red point

# Draw the second point
cv2.circle(image, point2, radius=5, color=(255, 0, 0), thickness=-1)  # Bl

cv2.imshow('Image with Points', image)
cv2.waitKey(0)  
cv2.destroyAllWindows()
