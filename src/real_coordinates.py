import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from typing import Tuple, List, Optional

class CoordinateTransformer:
    def __init__(self, camera_matrix: np.ndarray, dist_coefs: np.ndarray, pattern_size: Tuple[int, int]):
        self.camera_matrix = camera_matrix
        self.dist_coefs = dist_coefs
        self.pattern_size = pattern_size
        self.H = None
        self.H_inv = None
        self.corners = None

    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        """
        Undistort an image using the camera calibration parameters.

        Args:
            img: Input image to undistort

        Returns:
            Undistorted image
        """
        return cv2.undistort(img, self.camera_matrix, self.dist_coefs, None, self.camera_matrix)

    def compute_homography(self, img: np.ndarray) -> bool:
        """
        Compute the homography matrix from the image.

        Args:
            img: Input image

        Returns:
            Boolean indicating success
        """
        undistorted = self.undistort_image(img)
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, self.pattern_size)
        if not found:
            return False

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        grid = np.indices(self.pattern_size).T.reshape(-1, 2)
        world_points = np.zeros((grid.shape[0], 3), np.float32)
        world_points[:, :2] = grid * 1

        self.H, _ = cv2.findHomography(world_points[:, :2], self.corners.reshape(-1, 2))
        self.H_inv = np.linalg.inv(self.H)
        
        return True

    def pixel_to_world(self, pixel_points: np.ndarray) -> np.ndarray:
        """
        Convert pixel coordinates to world coordinates.

        Args:
            pixel_points: Array of pixel coordinates

        Returns:
            Array of world coordinates
        """
        if self.H_inv is None:
            raise ValueError("Homography matrix not computed. Call compute_homography first.")
        pixel_points_reshaped = pixel_points.reshape(-1, 1, 2)
        world_points = cv2.perspectiveTransform(pixel_points_reshaped, self.H_inv)
        return world_points.reshape(-1, 2)

    def get_coordinate_system_points(self, num_units: float = 2.0) -> Tuple[tuple, tuple, tuple]:
        """
        Get the points for drawing the coordinate system.

        Args:
            num_units: Length of the coordinate axes in world units

        Returns:
            Tuple of (origin, x_axis_end, y_axis_end) points
        """
        if self.corners is None:
            raise ValueError("Corners not detected. Run compute_homography first.")

        origin = tuple(self.corners[0].ravel().astype(int))

        x_axis_direction = np.array(self.corners[1].ravel()) - np.array(self.corners[0].ravel())
        one_unit_vector_x = (x_axis_direction) / 2.0

        y_axis_direction = np.array(self.corners[self.pattern_size[0]].ravel()) - np.array(self.corners[0].ravel())
        one_unit_vector_y = (y_axis_direction) / 2.0

        x_axis_end = (np.array(origin) + one_unit_vector_x * num_units).astype(int)
        y_axis_end = (np.array(origin) + one_unit_vector_y * num_units).astype(int)

        return origin, tuple(x_axis_end), tuple(y_axis_end)
    
    def create_world_coordinates_map(self, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create matrices containing the world coordinates for every pixel in the image.

        Args:
            image_shape: Tuple of (height, width) of the image

        Returns:
            Tuple of two 2D arrays (world_x, world_y) where each element contains
            the x and y world coordinates for the corresponding pixel
        """
        if self.H_inv is None:
            raise ValueError("Homography matrix not computed. Call compute_homography first.")

        # Create meshgrid of pixel coordinates
        height, width = image_shape[:2]
        y_coords, x_coords = np.mgrid[0:height, 0:width]

        # Reshape pixel coordinates into (N, 1, 2) array for perspectiveTransform
        pixel_coords = np.stack((x_coords, y_coords), axis=-1).astype(np.float32)
        pixel_coords_reshaped = pixel_coords.reshape(-1, 1, 2)

        # Transform all pixel coordinates to world coordinates
        world_coords = cv2.perspectiveTransform(pixel_coords_reshaped, self.H_inv)
        world_coords = world_coords.reshape(height, width, 2)

        # Split into separate x and y coordinate matrices
        world_x = world_coords[..., 0]
        world_y = world_coords[..., 1]

        return world_coords, world_x, world_y
    
    def visualize_coordinate_map(world_x: np.ndarray, world_y: np.ndarray, 
                           sample_step = 3, figsize = (10, 6)):
        """
        Visualize the world coordinate map using a grid of points.
        
        Args:
            world_x: 2D array of x world coordinates
            world_y: 2D array of y world coordinates
            sample_step: Number of pixels to skip between plotted points
            figsize: Size of the figure
        """
        figsize = tuple(map(float, figsize)) if isinstance(figsize, (list, tuple)) else (10, 6)

        plt.figure(figsize=figsize)
        sample_step = int(sample_step) if isinstance(sample_step, (int, float)) else 3

        # Sample points to avoid overcrowding the plot
        y_indices, x_indices = np.mgrid[0:world_x.shape[0]:sample_step, 
                                    0:world_x.shape[1]:sample_step]
        
        # Plot points
        plt.scatter(world_x[y_indices, x_indices], 
                world_y[y_indices, x_indices], 
                c='blue', alpha=0.5, s=1)
        
        # Add grid lines
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Add labels and title
        plt.xlabel('World X Coordinate')
        plt.ylabel('World Y Coordinate')
        plt.title('World Coordinate Map (Sampled Points)')
        
        # Make axes equal to preserve shape
        plt.axis('equal')
        plt.show()