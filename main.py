import numpy as np
import cv2
from src.cameracalibration import CameraCalibrator
from src.imagecalibration import CoordinateTransformer
def main():
    # Initialize transformer with your camera parameters
    camera_matrix = np.array([
        [618.4455161, 0., 392.65857132],
        [0., 616.38957715, 314.28598153],
        [0., 0., 1.]
    ])

    dist_coefs = np.array([0.40842381, -3.18499976, 0.01210313, -0.01222846, 7.79237145])

    # Create transformer
    transformer = CoordinateTransformer(camera_matrix, dist_coefs, (4, 4))

    # Load and process image
    img = cv2.imread(r'C:\Users\Yassine\Desktop\PIC\pic abdjelil\Calibration-image\images\checkerboard.jpeg')
    if img is None:
        raise IOError("Failed to load image.")

    # Compute homography
    if not transformer.compute_homography(img):
        raise ValueError("Failed to compute homography")
    # real world coordinates matrix
    world_coords, world_x, world_y = transformer.create_world_coordinates_map(img.shape)
    print(world_coords.shape)
    print(world_coords)
    # Print some example coordinates
    print("\nExample world coordinates at different pixel positions:")
    print(f"Top-left pixel (0,0) -> World coords: ({world_x[0,0]:.2f}, {world_y[0,0]:.2f})")
    print(f"Center pixel ({img.shape[0]//2},{img.shape[1]//2}) -> "
          f"World coords: ({world_x[img.shape[0]//2,img.shape[1]//2]:.2f}, "
          f"{world_y[img.shape[0]//2,img.shape[1]//2]:.2f})")

    # Transform example points
    pixel_points = np.array([[400, 370], [500, 400]], dtype=np.float32)
    world_points = transformer.pixel_to_world(pixel_points)

    # Visualize results
    undistorted = transformer.undistort_image(img)
    CoordinateTransformer.visualize_coordinate_map(undistorted, pixel_points, world_points, transformer)

if __name__ == "__main__":
    main()