import cv2
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

class PixelWorldMapper:
    def __init__(self, camera_matrix, dist_coeffs, known_distance, image, pattern_size, rvecs, tvecs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.known_distance = known_distance
        self.image = image
        self.pattern_size = pattern_size
        self.rvecs = rvecs.reshape(3, 1)  # Ensure rvecs is 3x1
        self.tvecs = tvecs.reshape(3, 1)  # Ensure tvecs is 3x1
        # self.scale_factor = self.calculate_scale_factor(pattern_size)
        self.undistorted_image = self.undistort_image()
        self.inv_camera_matrix = np.linalg.inv(self.camera_matrix)
        self.rotation_matrix = self.compute_rotation_matrix()
        
    def undistort_image(self):
        h, w = self.image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))

        
        return cv2.undistort(self.image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
    def compute_rotation_matrix(self):
        # Convertir rvecs en matrice de rotation
        rotation_matrix, _ = cv2.Rodrigues(self.rvecs)
        return rotation_matrix

    def pixel_to_normalized_coords(self, pixel_coords):

        pixel_homo = np.array([pixel_coords[0], pixel_coords[1], 1])
        normalized_coords = self.inv_camera_matrix @ pixel_homo
        return normalized_coords[:2]

    def normalized_to_world_coords(self, normalized_coords):
        """
        Convert normalized coordinates to world coordinates using the rotation matrix and translation vector.

        :param normalized_coords: Normalized coordinates (x', y').
        :return: World coordinates (X, Y, Z).
        """
        # Add a Z-coordinate (assume Z = 0 for simplicity)
        normalized_coords_homo = np.append(normalized_coords, 1)  # Convert to 3D (x', y', 0)
        
        # Transform to world coordinates
        world_coords = self.rotation_matrix @ normalized_coords_homo + self.tvecs.flatten()
        
        return world_coords[:2]  # Return only X and Y (ignore Z for 2D mapping)

    def pixel_to_world_coords(self, pixel_coords):

        normalized_coords = self.pixel_to_normalized_coords(pixel_coords)
        return self.normalized_to_world_coords(normalized_coords)

    
    def compute_world_coordinates_matrix(self):
        """
        Compute the world coordinates for every pixel in the undistorted image.

        :return: World coordinates matrix (height x width x 2).
        """
        height, width = self.undistorted_image.shape[:2]
        world_coords = np.zeros((height, width, 2), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                # Convert pixel coordinates to normalized coordinates
                normalized_coords = self.pixel_to_normalized_coords((x, y))
                
                # Convert normalized coordinates to world coordinates
                world_coords[y, x] = self.normalized_to_world_coords(normalized_coords)

        return world_coords
    
    def calculate_point_distance(self, world_coords_matrix, point1, point2):
        # Obtenir les coordonnées du monde pour les deux points
        world_point1 = world_coords_matrix[point1[1], point1[0]]
        world_point2 = world_coords_matrix[point2[1], point2[0]]
        
        # Calculer la distance Euclidienne
        distance = np.linalg.norm(world_point2 - world_point1)
        
        return distance

    def distance_between_pixels(self, point1, point2):
        world_coords_matrix = self.compute_world_coordinates_matrix()
        return self.calculate_point_distance(world_coords_matrix, point1, point2)