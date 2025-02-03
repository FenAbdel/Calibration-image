import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np

# Adjust these imports as needed depending on your project structure.
from cameraCalibration import CameraCalibrator
from realCordinates import CoordinateTransformer

class CalibrationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Camera Calibration & Coordinate Transformer")
        self.geometry("400x250")

        # Parameter entries
        tk.Label(self, text="Square Size (mm):").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.square_size_entry = tk.Entry(self)
        self.square_size_entry.insert(0, "25.0")
        self.square_size_entry.grid(row=0, column=1, padx=10, pady=10)

        tk.Label(self, text="Pattern Size (e.g. 9,6):").grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.pattern_size_entry = tk.Entry(self)
        self.pattern_size_entry.insert(0, "9,6")
        self.pattern_size_entry.grid(row=1, column=1, padx=10, pady=10)

        # Buttons for image selection
        self.calib_button = tk.Button(self, text="Select Calibration Images", command=self.select_calibration_images)
        self.calib_button.grid(row=2, column=0, columnspan=2, pady=5)

        self.test_button = tk.Button(self, text="Select Test Image", command=self.select_test_image)
        self.test_button.grid(row=3, column=0, columnspan=2, pady=5)

        # Button to run calibration and transformation
        self.run_button = tk.Button(self, text="Calibrate & Compute Coordinates", command=self.run_calibration)
        self.run_button.grid(row=4, column=0, columnspan=2, pady=10)

        # Variables to store file paths
        self.calibration_files = None
        self.test_image_path = None

    def select_calibration_images(self):
        self.calibration_files = filedialog.askopenfilenames(
            title="Select Calibration Images", 
            filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")]
        )
        if self.calibration_files:
            messagebox.showinfo("Images Selected", f"{len(self.calibration_files)} calibration images selected.")

    def select_test_image(self):
        self.test_image_path = filedialog.askopenfilename(
            title="Select Test Image", 
            filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")]
        )
        if self.test_image_path:
            messagebox.showinfo("Image Selected", f"Test image selected:\n{self.test_image_path}")

    def run_calibration(self):
        # Read parameter values
        try:
            square_size = float(self.square_size_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid square size. Please enter a valid number.")
            return

        pattern_size_str = self.pattern_size_entry.get().strip()
        try:
            pattern_size = tuple(map(int, pattern_size_str.split(',')))
        except:
            messagebox.showerror("Error", "Invalid pattern size. Use format: 9,6")
            return

        if not self.calibration_files or not self.test_image_path:
            messagebox.showerror("Error", "Please select calibration images and a test image.")
            return

        # Perform camera calibration using the selected calibration images.
        calibrator = CameraCalibrator(square_size, pattern_size)
        obj_points = []
        img_points = []
        for i, file in enumerate(self.calibration_files):
            success, corners = calibrator._process_image(file, i, visualize=False)
            if success:
                img_points.append(corners)
                obj_points.append(calibrator._pattern_points)

        if len(obj_points) < 1:
            messagebox.showerror("Error", "No valid calibration images (chessboard not detected).")
            return

        # Use the first calibration image to determine the image size.
        img = cv2.imread(self.calibration_files[0])
        h, w = img.shape[:2]
        ret, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, (w, h), None, None
        )
        # (ret is the RMS error)

        # Load the test image
        test_img = cv2.imread(self.test_image_path)
        if test_img is None:
            messagebox.showerror("Error", "Unable to load the test image.")
            return

        # Compute homography and undistort the test image.
        transformer = CoordinateTransformer(camera_matrix, dist_coefs, pattern_size)
        if not transformer.compute_homography(test_img):
            messagebox.showerror("Error", "Failed to compute homography for the test image.")
            return
        undistorted = transformer.undistort_image(test_img)

        # Draw the coordinate system on the image.
        origin, x_axis_end, y_axis_end = transformer.get_coordinate_system_points(num_units=2.0)
        cv2.arrowedLine(undistorted, origin, x_axis_end, (255, 0, 0), 2)  # X-axis in blue
        cv2.arrowedLine(undistorted, origin, y_axis_end, (0, 255, 0), 2)  # Y-axis in green
        cv2.circle(undistorted, origin, 5, (0, 0, 255), -1)              # Origin in red

        # Randomly select a pixel within the undistorted image.
        height, width = undistorted.shape[:2]
        rand_x = np.random.randint(0, width)
        rand_y = np.random.randint(0, height)
        random_pixel = np.array([[rand_x, rand_y]], dtype=np.float32)
        world_coord = transformer.pixel_to_world(random_pixel)

        # Draw and annotate the random point.
        cv2.circle(undistorted, (rand_x, rand_y), 5, (0, 255, 255), -1)  # Yellow circle
        coord_text = f"({world_coord[0,0]:.2f}, {world_coord[0,1]:.2f})"
        cv2.putText(undistorted, coord_text, (rand_x + 10, rand_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the result in a new window.
        cv2.imshow("Result", undistorted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        messagebox.showinfo("Done", f"Random Pixel: ({rand_x}, {rand_y})\nWorld Coordinates: {coord_text}")

if __name__ == "__main__":
    app = CalibrationApp()
    app.mainloop()
