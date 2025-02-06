# src/gui/map_frame.py
import os
import time
import threading
import tkinter as tk
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import qrcode
from PIL import Image, ImageTk

from calibration_db import load_calibrations
from real_coordinates import CoordinateTransformer
from capture_server import start_capture_server_in_thread, get_local_ip

# Compute the project root (three levels up) so that the photos folder is at the project root.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHOTOS_FOLDER = os.path.join(BASE_DIR, "photos")

class MapFrame(tb.Frame):
    def __init__(self, master):
        super().__init__(master, padding=20)
        self.grid(sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)  # Text output area expands
        
        # ---- Section 1: Calibration Options ----
        options_frame = tb.Labelframe(self, text="Calibration Option", bootstyle="info")
        options_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        options_frame.columnconfigure(1, weight=1)
        tb.Label(options_frame, text="Use Existing Calibration:", font=("Segoe UI", 12))\
            .grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.use_existing_var = tb.StringVar(value="yes")
        tb.Radiobutton(options_frame, text="Yes", variable=self.use_existing_var, value="yes", command=self.update_camera_choices)\
            .grid(row=0, column=1, sticky="w", padx=5)
        tb.Radiobutton(options_frame, text="No (Calibrate New)", variable=self.use_existing_var, value="no", command=self.update_camera_choices)\
            .grid(row=0, column=2, sticky="w", padx=5)
        tb.Label(options_frame, text="Camera Name:", font=("Segoe UI", 12))\
            .grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.camera_choice = tb.Combobox(options_frame, bootstyle="info")
        self.camera_choice.grid(row=1, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.update_camera_choices()
        
        # ---- Section 2: Test Image Selection/Capture ----
        test_frame = tb.Labelframe(self, text="Test Image", bootstyle="primary")
        test_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        test_frame.columnconfigure((0, 1), weight=1)
        tb.Button(test_frame, text="Select Test Image", command=self.select_test_image, bootstyle="primary-outline")\
            .grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        tb.Button(test_frame, text="Capture Test Image from Phone", command=self.capture_test_image, bootstyle="success-outline")\
            .grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        # ---- Section 3: Action Buttons ----
        action_frame = tb.Frame(self)
        action_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        action_frame.columnconfigure((0, 1), weight=1)
        tb.Button(action_frame, text="Generate Coordinate Map", command=self.generate_map, bootstyle="warning")\
            .grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        tb.Button(action_frame, text="Export Matrix to TXT", command=self.export_matrix, bootstyle="secondary")\
            .grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        # ---- Section 4: Text Output Area ----
        self.text = tk.Text(self, width=70, height=15, font=("Segoe UI", 10))
        self.text.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)
        
        self.test_image_path = None
        self.last_world_coords = None  # For exporting the coordinate matrix

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

    def capture_test_image(self):
        """
        Launch the capture server, display a QR code, and automatically select the newly captured image.
        """
        start_capture_server_in_thread()
        local_ip = get_local_ip()
        url = f"https://{local_ip}:5000"
        # Create QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=4)
        qr.add_data(url)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_img_tk = ImageTk.PhotoImage(qr_img)
        qr_window = tb.Toplevel(self)
        qr_window.title("Capture Test Image from Phone")
        tb.Label(qr_window, text="Scan this QR code with your phone to capture a test image:", font=("Segoe UI", 12, "bold"))\
            .grid(row=0, column=0, padx=10, pady=10)
        tb.Label(qr_window, image=qr_img_tk).grid(row=1, column=0, padx=10, pady=10)
        tb.Label(qr_window, text=url, font=("Segoe UI", 12)).grid(row=2, column=0, padx=10, pady=10)
        qr_window.qr_img_tk = qr_img_tk  # Keep reference
        
        messagebox.showinfo("Capture Test Image", 
            "After scanning, capture the image with your phone. This window will close automatically when a new image is detected.")
        
        # Record baseline files in the photos folder
        baseline = set(os.listdir(PHOTOS_FOLDER)) if os.path.exists(PHOTOS_FOLDER) else set()
        
        def wait_for_new_image():
            timeout = 30  # seconds
            start_time = time.time()
            new_image = None
            while time.time() - start_time < timeout:
                time.sleep(1)
                if os.path.exists(PHOTOS_FOLDER):
                    current_files = set(os.listdir(PHOTOS_FOLDER))
                    new_files = current_files - baseline
                    if new_files:
                        full_paths = [os.path.join(PHOTOS_FOLDER, f) for f in new_files]
                        new_image = max(full_paths, key=os.path.getmtime)
                        break
            if new_image:
                self.test_image_path = new_image
                self.after(0, lambda: messagebox.showinfo("Test Image Captured", f"New test image captured and selected:\n{new_image}"))
                self.after(0, qr_window.destroy)
            else:
                self.after(0, lambda: messagebox.showerror("Timeout", "No new test image detected within 30 seconds."))
                self.after(0, qr_window.destroy)
        
        threading.Thread(target=wait_for_new_image, daemon=True).start()

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

        try:
            world_coords, world_x, world_y = transformer.create_world_coordinates_map(test_img.shape)
            self.last_world_coords = world_coords  # Save for export
        except Exception as e:
            messagebox.showerror("Error", f"Error creating coordinate map: {e}")
            return

        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, "World Coordinates Map (H x W x 2):\n")
        self.text.insert(tk.END, str(world_coords))
        self.text.insert(tk.END, "\n\nWorld X Coordinates (H x W):\n")
        self.text.insert(tk.END, str(world_x))
        self.text.insert(tk.END, "\n\nWorld Y Coordinates (H x W):\n")
        self.text.insert(tk.END, str(world_y))
        messagebox.showinfo("Map Generated", "Coordinate map generated successfully.")

    def export_matrix(self):
        if self.last_world_coords is None:
            messagebox.showerror("Error", "No coordinate map available to export. Generate it first.")
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            title="Save Coordinate Matrix"
        )
        if not filepath:
            return
        try:
            # Convert the entire matrix to a string without truncation.
            # np.array2string with threshold=np.inf ensures that all values are printed.
            array_str = np.array2string(self.last_world_coords, threshold=np.inf, separator=', ')
            with open(filepath, 'w') as f:
                f.write(array_str)
            messagebox.showinfo("Export Successful", f"Coordinate matrix exported to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export matrix: {e}")

