# main.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import json, os

from cameraCalibration import CameraCalibrator
from realCordinates import CoordinateTransformer

# --- Calibration Database Helper Functions ---
CALIB_DB = "calibration_db.json"

def load_calibrations():
    if os.path.exists(CALIB_DB):
        with open(CALIB_DB, "r") as f:
            return json.load(f)
    else:
        return {}

def save_calibration(camera_name, camera_matrix, dist_coefs, square_size, pattern_size):
    calibrations = load_calibrations()
    calibrations[camera_name] = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coefs": dist_coefs.tolist(),
        "square_size": square_size,
        "pattern_size": list(pattern_size)
    }
    with open(CALIB_DB, "w") as f:
        json.dump(calibrations, f, indent=4)

# --- Calibration Tab ---
class CalibrationFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        tk.Label(self, text="Camera Name:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.camera_name_entry = tk.Entry(self)
        self.camera_name_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(self, text="Square Size (mm):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.square_size_entry = tk.Entry(self)
        self.square_size_entry.insert(0, "25.0")
        self.square_size_entry.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(self, text="Pattern Size (e.g. 9,6):").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.pattern_size_entry = tk.Entry(self)
        self.pattern_size_entry.insert(0, "9,6")
        self.pattern_size_entry.grid(row=2, column=1, padx=5, pady=5)

        self.select_calib_btn = tk.Button(self, text="Select Calibration Images", command=self.select_calib_images)
        self.select_calib_btn.grid(row=3, column=0, columnspan=2, pady=5)

        self.calibrate_btn = tk.Button(self, text="Calibrate Camera", command=self.calibrate_camera)
        self.calibrate_btn.grid(row=4, column=0, columnspan=2, pady=5)

        self.calib_files = None

    def select_calib_images(self):
        self.calib_files = filedialog.askopenfilenames(
            title="Select Calibration Images",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
        )
        if self.calib_files:
            messagebox.showinfo("Images Selected", f"{len(self.calib_files)} images selected.")

    def calibrate_camera(self):
        camera_name = self.camera_name_entry.get().strip()
        if not camera_name:
            messagebox.showerror("Error", "Please enter a camera name.")
            return
        try:
            square_size = float(self.square_size_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid square size.")
            return
        try:
            pattern_size = tuple(map(int, self.pattern_size_entry.get().split(',')))
        except:
            messagebox.showerror("Error", "Invalid pattern size. Use format: 9,6")
            return

        if not self.calib_files:
            messagebox.showerror("Error", "Please select calibration images.")
            return

        calibrator = CameraCalibrator(square_size, pattern_size)
        obj_points = []
        img_points = []
        for i, file in enumerate(self.calib_files):
            success, corners = calibrator._process_image(file, i, visualize=False)
            if success:
                img_points.append(corners)
                obj_points.append(calibrator._pattern_points)

        if len(obj_points) < 1:
            messagebox.showerror("Error", "Calibration failed: no valid images.")
            return

        img = cv2.imread(self.calib_files[0])
        h, w = img.shape[:2]
        ret, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, (w, h), None, None
        )
        save_calibration(camera_name, camera_matrix, dist_coefs, square_size, pattern_size)
        messagebox.showinfo("Success", f"Camera '{camera_name}' calibrated and saved.")

# --- Real-World Coordinates Tab ---
class CoordinatesFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.use_existing_var = tk.StringVar(value="yes")
        tk.Label(self, text="Use Existing Calibration:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tk.Radiobutton(self, text="Yes", variable=self.use_existing_var, value="yes", command=self.update_camera_choices).grid(row=0, column=1)
        tk.Radiobutton(self, text="No (Calibrate New)", variable=self.use_existing_var, value="no", command=self.update_camera_choices).grid(row=0, column=2)

        tk.Label(self, text="Camera Name:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.camera_choice = ttk.Combobox(self, values=[])
        self.camera_choice.grid(row=1, column=1, columnspan=2, padx=5, pady=5)
        self.update_camera_choices()

        self.select_test_btn = tk.Button(self, text="Select Test Image", command=self.select_test_image)
        self.select_test_btn.grid(row=2, column=0, columnspan=3, pady=5)

        self.calc_coords_btn = tk.Button(self, text="Calculate Coordinates", command=self.calculate_coordinates)
        self.calc_coords_btn.grid(row=3, column=0, columnspan=3, pady=5)

        self.test_image_path = None

    def update_camera_choices(self):
        calibrations = load_calibrations()
        cam_names = list(calibrations.keys())
        self.camera_choice['values'] = cam_names

    def select_test_image(self):
        self.test_image_path = filedialog.askopenfilename(
            title="Select Test Image",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
        )
        if self.test_image_path:
            messagebox.showinfo("Test Image", f"Test image selected:\n{self.test_image_path}")

    def calculate_coordinates(self):
        if self.use_existing_var.get() == "yes":
            calibrations = load_calibrations()
            camera_name = self.camera_choice.get().strip()
            if not camera_name or camera_name not in calibrations:
                messagebox.showerror("Error", "Please select a valid camera from existing calibrations.")
                return
            cam_data = calibrations[camera_name]
            camera_matrix = np.array(cam_data["camera_matrix"])
            dist_coefs = np.array(cam_data["dist_coefs"])
            pattern_size = tuple(cam_data["pattern_size"])
        else:
            messagebox.showinfo("Info", "Please calibrate a new camera in the Calibration tab first.")
            return

        if not self.test_image_path:
            messagebox.showerror("Error", "Please select a test image.")
            return

        test_img = cv2.imread(self.test_image_path)
        if test_img is None:
            messagebox.showerror("Error", "Could not load test image.")
            return

        transformer = CoordinateTransformer(camera_matrix, dist_coefs, pattern_size)
        if not transformer.compute_homography(test_img):
            messagebox.showerror("Error", "Failed to compute homography on test image.")
            return
        undistorted = transformer.undistort_image(test_img)

        # For demonstration, randomly pick a pixel:
        h_img, w_img = undistorted.shape[:2]
        rand_x = np.random.randint(0, w_img)
        rand_y = np.random.randint(0, h_img)
        random_pixel = np.array([[rand_x, rand_y]], dtype=np.float32)
        world_coord = transformer.pixel_to_world(random_pixel)

        origin, x_axis_end, y_axis_end = transformer.get_coordinate_system_points(num_units=2.0)
        cv2.arrowedLine(undistorted, origin, x_axis_end, (255, 0, 0), 2)
        cv2.arrowedLine(undistorted, origin, y_axis_end, (0, 255, 0), 2)
        cv2.circle(undistorted, origin, 5, (0, 0, 255), -1)
        cv2.circle(undistorted, (rand_x, rand_y), 5, (0, 255, 255), -1)
        coord_text = f"({world_coord[0,0]:.2f}, {world_coord[0,1]:.2f})"
        cv2.putText(undistorted, coord_text, (rand_x + 10, rand_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow("Result", undistorted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        messagebox.showinfo("Coordinates", f"Random Pixel: ({rand_x}, {rand_y})\nWorld Coordinates: {coord_text}")

# --- Coordinate Map Tab ---
class MapFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        # This tab displays the full coordinate map arrays.
        tk.Label(self, text="Use Existing Calibration:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.use_existing_var = tk.StringVar(value="yes")
        tk.Radiobutton(self, text="Yes", variable=self.use_existing_var, value="yes", command=self.update_camera_choices).grid(row=0, column=1)
        tk.Radiobutton(self, text="No (Calibrate New)", variable=self.use_existing_var, value="no", command=self.update_camera_choices).grid(row=0, column=2)

        tk.Label(self, text="Camera Name:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.camera_choice = ttk.Combobox(self, values=[])
        self.camera_choice.grid(row=1, column=1, columnspan=2, padx=5, pady=5)
        self.update_camera_choices()

        self.select_test_btn = tk.Button(self, text="Select Test Image", command=self.select_test_image)
        self.select_test_btn.grid(row=2, column=0, columnspan=3, pady=5)

        self.gen_map_btn = tk.Button(self, text="Generate Coordinate Map", command=self.generate_map)
        self.gen_map_btn.grid(row=3, column=0, columnspan=3, pady=5)

        self.text = tk.Text(self, width=70, height=15)
        self.text.grid(row=4, column=0, columnspan=3, padx=5, pady=5)

        self.test_image_path = None

    def update_camera_choices(self):
        calibrations = load_calibrations()
        cam_names = list(calibrations.keys())
        self.camera_choice['values'] = cam_names

    def select_test_image(self):
        self.test_image_path = filedialog.askopenfilename(
            title="Select Test Image",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
        )
        if self.test_image_path:
            messagebox.showinfo("Test Image", f"Test image selected:\n{self.test_image_path}")

    def generate_map(self):
        if self.use_existing_var.get() == "yes":
            calibrations = load_calibrations()
            camera_name = self.camera_choice.get().strip()
            if not camera_name or camera_name not in calibrations:
                messagebox.showerror("Error", "Please select a valid camera from existing calibrations.")
                return
            cam_data = calibrations[camera_name]
            camera_matrix = np.array(cam_data["camera_matrix"])
            dist_coefs = np.array(cam_data["dist_coefs"])
            pattern_size = tuple(cam_data["pattern_size"])
        else:
            messagebox.showinfo("Info", "Please calibrate a new camera in the Calibration tab first.")
            return

        if not self.test_image_path:
            messagebox.showerror("Error", "Please select a test image.")
            return

        test_img = cv2.imread(self.test_image_path)
        if test_img is None:
            messagebox.showerror("Error", "Could not load test image.")
            return

        transformer = CoordinateTransformer(camera_matrix, dist_coefs, pattern_size)
        if not transformer.compute_homography(test_img):
            messagebox.showerror("Error", "Failed to compute homography on test image.")
            return

        # Call the provided function to create the world coordinate map.
        try:
            world_coords, world_x, world_y = transformer.create_world_coordinates_map(test_img.shape)
        except Exception as e:
            messagebox.showerror("Error", f"Error creating coordinate map: {e}")
            return

        # Clear previous content and display the matrices.
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, "World Coordinates Map (H x W x 2):\n")
        self.text.insert(tk.END, str(world_coords))
        self.text.insert(tk.END, "\n\nWorld X Coordinates (H x W):\n")
        self.text.insert(tk.END, str(world_x))
        self.text.insert(tk.END, "\n\nWorld Y Coordinates (H x W):\n")
        self.text.insert(tk.END, str(world_y))

# --- Main Application Window ---
class CalibrationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Camera Calibration & Real-World Coordinates")
        self.geometry("600x500")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.calibration_frame = CalibrationFrame(notebook)
        self.coordinates_frame = CoordinatesFrame(notebook)
        self.map_frame = MapFrame(notebook)

        notebook.add(self.calibration_frame, text="Calibration")
        notebook.add(self.coordinates_frame, text="Coordinates")
        notebook.add(self.map_frame, text="Coordinate Map")

if __name__ == "__main__":
    app = CalibrationApp()
    app.mainloop()
