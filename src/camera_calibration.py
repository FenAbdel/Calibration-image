import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from typing import Tuple, List, Optional

class CameraCalibrator:
    def __init__(self, square_size: float, pattern_size: Tuple[int, int]):
        self.square_size = square_size
        self.pattern_size = pattern_size
        self.camera_matrix = None
        self.dist_coefs = None
        self._pattern_points = self._create_pattern_points()

    def _create_pattern_points(self) -> np.ndarray:
        pattern_points = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)
        pattern_points *= self.square_size
        return pattern_points

    def calibrate_from_images(self, img_mask: str, visualize: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
        img_names = glob(img_mask)
        obj_points = []
        img_points = []

        if visualize:
            plt.figure(figsize=(20, 20))

        for i, fn in enumerate(img_names):
            success, corners = self._process_image(fn, i, visualize)
            if success:
                img_points.append(corners)
                obj_points.append(self._pattern_points)

        if visualize:
            plt.show()

        h, w = cv2.imread(img_names[0]).shape[:2]
        rms, self.camera_matrix, self.dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(
            obj_points, img_points, (w, h), None, None
        )
        return rms, self.camera_matrix, self.dist_coefs

    def _process_image(self, img_path: str, index: int, visualize: bool) -> Tuple[bool, Optional[np.ndarray]]:
        print(f"Processing {img_path}...")
        imgBGR = cv2.imread(img_path)
        if imgBGR is None:
            print(f"Failed to load {img_path}")
            return False, None

        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)

        found, corners = cv2.findChessboardCorners(img, self.pattern_size)
        if not found:
            print("Chessboard not found")
            return False, None

        if visualize and index < 12:
            img_w_corners = cv2.drawChessboardCorners(imgRGB, self.pattern_size, corners, found)
            plt.subplot(4, 3, index + 1)
            plt.imshow(img_w_corners)

        print(f"{img_path}... OK")
        return True, corners.reshape(-1, 2)
