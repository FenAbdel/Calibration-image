# src/gui/coordinates_frame.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import matplotlib.pyplot as plt

from calibration_db import load_calibrations
from real_coordinates import CoordinateTransformer  # Your existing coordinate transformation code

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

        tk.Label(self, text="Pixel Selection Mode:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.pixel_mode_var = tk.StringVar(value="manual")
        tk.Radiobutton(self, text="Manual Entry", variable=self.pixel_mode_var, value="manual", command=self.toggle_manual_entries).grid(row=3, column=1, padx=5, pady=5)
        tk.Radiobutton(self, text="Click on Image", variable=self.pixel_mode_var, value="click", command=self.toggle_manual_entries).grid(row=3, column=2, padx=5, pady=5)

        self.manual_frame = tk.Frame(self)
        tk.Label(self.manual_frame, text="Pixel X:").grid(row=0, column=0, padx=5, pady=5)
        self.pixel_x_entry = tk.Entry(self.manual_frame, width=5)
        self.pixel_x_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(self.manual_frame, text="Pixel Y:").grid(row=0, column=2, padx=5, pady=5)
        self.pixel_y_entry = tk.Entry(self.manual_frame, width=5)
        self.pixel_y_entry.grid(row=0, column=3, padx=5, pady=5)
        self.manual_frame.grid(row=4, column=0, columnspan=3, pady=5)

        self.calc_coords_btn = tk.Button(self, text="Calculate Coordinates", command=self.calculate_coordinates)
        self.calc_coords_btn.grid(row=5, column=0, columnspan=3, pady=5)

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

    def toggle_manual_entries(self):
        if self.pixel_mode_var.get() == "manual":
            self.manual_frame.grid()
        else:
            self.manual_frame.grid_remove()

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
        if self.pixel_mode_var.get() == "manual":
            try:
                x = int(self.pixel_x_entry.get())
                y = int(self.pixel_y_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter valid integer values for pixel coordinates.")
                return
            selected_pixel = np.array([[x, y]], dtype=np.float32)
        else:
            plt.figure("Select Pixel - Click on the image")
            undistorted_rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
            plt.imshow(undistorted_rgb)
            plt.title("Click one point on the image")
            pts = plt.ginput(1, timeout=0)
            plt.close()
            if not pts:
                messagebox.showerror("Error", "No point was selected.")
                return
            x, y = pts[0]
            selected_pixel = np.array([[int(x), int(y)]], dtype=np.float32)

        try:
            world_coord = transformer.pixel_to_world(selected_pixel)
        except Exception as e:
            messagebox.showerror("Error", f"Error calculating world coordinate: {e}")
            return

        origin, x_axis_end, y_axis_end = transformer.get_coordinate_system_points(num_units=2.0)
        cv2.arrowedLine(undistorted, origin, x_axis_end, (255, 0, 0), 2)
        cv2.arrowedLine(undistorted, origin, y_axis_end, (0, 255, 0), 2)
        cv2.circle(undistorted, origin, 5, (0, 0, 255), -1)
        cv2.circle(undistorted, (int(selected_pixel[0,0]), int(selected_pixel[0,1])), 5, (0, 255, 255), -1)
        coord_text = f"({world_coord[0,0]:.2f}, {world_coord[0,1]:.2f})"
        cv2.putText(undistorted, coord_text, (int(selected_pixel[0,0]) + 10, int(selected_pixel[0,1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow("Result", undistorted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        messagebox.showinfo("Coordinates", f"Selected Pixel: ({int(selected_pixel[0,0])}, {int(selected_pixel[0,1])})\nWorld Coordinates: {coord_text}")