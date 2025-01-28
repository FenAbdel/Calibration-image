import cv2
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

class CalibrateCamera:
    def __init__(self, img_mask,figsize, square_size, pattern_size):
        self.img_mask = img_mask
        self.figsize = figsize
        self.square_size = square_size
        self.pattern_size = pattern_size
    
    
    def calibrate(self):

        img_names = glob(self.img_mask)
        num_images = len(img_names)

        pattern_points = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)
        pattern_points *= self.square_size

        obj_points = []
        img_points = []
        h, w = cv2.imread(img_names[0]).shape[:2]

        plt.figure(figsize=self.figsize)

        for i, fn in enumerate(img_names):
            print("processing %s... " % fn)
            imgBGR = cv2.imread(fn)

            if imgBGR is None:
                print("Failed to load", fn)
                continue

            imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)
            print(f"Expected size: {w} x {h}")
            print(f"Actual size: {img.shape[1]} x {img.shape[0]}")

            assert (
                w == img.shape[1] and h == img.shape[0]
            ), f"size: {img.shape[1]} x {img.shape[0]}"
            found, corners = cv2.findChessboardCorners(img, self.pattern_size)

            if not found:
                print("chessboard not found")
                continue

            if i < 12:
                img_w_corners = cv2.drawChessboardCorners(imgRGB, self.pattern_size, corners, found)
                plt.subplot(4, 3, i + 1)
                plt.imshow(img_w_corners)

            print(f"{fn}... OK")
            img_points.append(corners.reshape(-1, 2))
            obj_points.append(pattern_points)
        plt.show()
        
        rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(
            obj_points, img_points, (w, h), None, None
        )

        return rms, camera_matrix, dist_coefs, _rvecs, _tvecs

