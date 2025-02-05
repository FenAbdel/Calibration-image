# src/gui/calibration_frame.py
import os
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import qrcode
from PIL import Image, ImageTk
import threading

from calibration_db import save_calibration, load_calibrations
from capture_server import start_capture_server_in_thread, get_local_ip
from camera_calibration import CameraCalibrator  # Your existing calibration code

# Compute the project root (three levels up) so that the photos folder is at the project root.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHOTOS_FOLDER = os.path.join(BASE_DIR, "photos")

class CalibrationFrame(tb.Frame):
    def __init__(self, master):
        super().__init__(master, padding=20)
        # Make sure this frame expands to fill its container.
        self.grid(sticky="nsew")
        self.columnconfigure(0, weight=1)
        # We'll use grid layout for all internal components.
        # ---------------------------
        # 1. Header: Camera Options
        header = tb.Labelframe(self, text="Camera Options", bootstyle="info")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        header.columnconfigure(1, weight=1)
        tb.Label(header, text="Camera Name:", font=("Segoe UI", 12)).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.camera_name_entry = tb.Entry(header, font=("Segoe UI", 12))
        self.camera_name_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        tb.Label(header, text="Square Size (mm):", font=("Segoe UI", 12)).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.square_size_entry = tb.Entry(header, font=("Segoe UI", 12))
        self.square_size_entry.insert(0, "30.0")
        self.square_size_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        tb.Label(header, text="Pattern Size (e.g. 4,4):", font=("Segoe UI", 12)).grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.pattern_size_entry = tb.Entry(header, font=("Segoe UI", 12))
        self.pattern_size_entry.insert(0, "4,4")
        self.pattern_size_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        
        # ---------------------------
        # 2. Actions: Buttons for selecting/capturing images
        actions = tb.Frame(self)
        actions.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        # Configure three equal-weight columns.
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        actions.columnconfigure(2, weight=1)
        tb.Button(actions, text="Select Calibration Images", command=self.select_calib_images, bootstyle="primary")\
            .grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        tb.Button(actions, text="Capture Images from Phone", command=self.start_capture_server, bootstyle="success")\
            .grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        tb.Button(actions, text="Refresh Captured Images", command=self.load_captured_images, bootstyle="warning")\
            .grid(row=0, column=2, sticky="ew", padx=5, pady=5)
        
        # ---------------------------
        # 3. Thumbnails Area: Captured images appear here.
        thumb_frame = tb.Labelframe(self, text="Captured Images (click to validate)", bootstyle="info")
        thumb_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        # Let this frame expand.
        self.rowconfigure(2, weight=1)
        thumb_frame.columnconfigure(0, weight=1)
        thumb_frame.rowconfigure(0, weight=1)
        self.images_canvas = tb.Canvas(thumb_frame, background="#ffffff", bd=0, highlightthickness=0)
        self.images_canvas.grid(row=0, column=0, sticky="nsew")
        self.hscroll = tb.Scrollbar(thumb_frame, orient="horizontal", command=self.images_canvas.xview)
        self.hscroll.grid(row=1, column=0, sticky="ew", padx=5, pady=(0,5))
        self.images_canvas.configure(xscrollcommand=self.hscroll.set)
        self.images_frame = tb.Frame(self.images_canvas)
        self.images_canvas.create_window((0, 0), window=self.images_frame, anchor="nw")
        self.images_frame.bind("<Configure>", lambda e: self.images_canvas.configure(scrollregion=self.images_canvas.bbox("all")))
        
        # ---------------------------
        # 4. Status and Calibrate Button at the bottom.
        self.status_label = tb.Label(self, text="0 images validated for calibration.", font=("Segoe UI", 12))
        self.status_label.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
        tb.Button(self, text="Calibrate Camera", command=self.calibrate_camera, bootstyle="success-outline")\
            .grid(row=4, column=0, sticky="ew", padx=10, pady=10)
        
        # Internal state variables
        self.calib_files = []  # List of validated image file paths
        self.thumbnail_images = {}  # Keep references to PhotoImage objects

    def select_calib_images(self):
        """Allow manual selection of images for calibration."""
        files = filedialog.askopenfilenames(
            title="Select Calibration Images",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
        )
        if files:
            self.calib_files.extend(files)
            messagebox.showinfo("Images Selected", f"{len(files)} images added.")
            self.update_status_label()

    def load_captured_images(self):
        """Load thumbnails from the PHOTOS_FOLDER into the thumbnails area horizontally."""
        # Clear previous thumbnails.
        for widget in self.images_frame.winfo_children():
            widget.destroy()
        self.thumbnail_images.clear()
        
        if os.path.exists(PHOTOS_FOLDER):
            files = sorted(os.listdir(PHOTOS_FOLDER))
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                # Place all thumbnails in row 0, incrementing the column index.
                for idx, filename in enumerate(image_files):
                    full_path = os.path.join(PHOTOS_FOLDER, filename)
                    try:
                        img = Image.open(full_path)
                        img.thumbnail((100, 100))
                        photo = ImageTk.PhotoImage(img)
                        self.thumbnail_images[full_path] = photo  # Keep a reference so it isn’t garbage-collected.
                        btn = tb.Button(self.images_frame, image=photo, command=lambda fp=full_path: self.validate_image(fp), bootstyle="link")
                        btn.grid(row=0, column=idx, padx=5, pady=5)
                    except Exception as e:
                        print("Error loading image:", full_path, e)
            else:
                messagebox.showinfo("Load Images", "No captured images found in the photos folder.")
        else:
            messagebox.showerror("Error", f"Photos folder not found at {PHOTOS_FOLDER}.")


    def validate_image(self, full_path):
        """Add the clicked image to the validated list (if not already added)."""
        if full_path not in self.calib_files:
            self.calib_files.append(full_path)
            messagebox.showinfo("Image Validated", "Image validated for calibration.")
            self.update_status_label()
        else:
            messagebox.showwarning("Already Selected", "This image has already been validated.")

    def update_status_label(self):
        self.status_label.config(text=f"{len(self.calib_files)} images validated for calibration.")

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
        except Exception:
            messagebox.showerror("Error", "Invalid pattern size. Use format: 4,4")
            return
        if not self.calib_files:
            messagebox.showerror("Error", "No images validated for calibration.")
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
        # Use the dimensions of the first image.
        img = cv2.imread(self.calib_files[0])
        h, w = img.shape[:2]
        ret, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, (w, h), None, None
        )
        save_calibration(camera_name, camera_matrix, dist_coefs, square_size, pattern_size)
        messagebox.showinfo("Success", f"Camera '{camera_name}' calibrated and saved.")

    def start_capture_server(self):
        start_capture_server_in_thread()
        local_ip = get_local_ip()
        url = f"https://{local_ip}:5000"
        qr = qrcode.QRCode(version=1, box_size=10, border=4)
        qr.add_data(url)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_img_tk = ImageTk.PhotoImage(qr_img)
        qr_window = tb.Toplevel(self)
        qr_window.title("Scan QR Code to Capture Images")
        tb.Label(qr_window, text="Scan this QR code with your phone to capture calibration images:", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, padx=10, pady=10)
        tb.Label(qr_window, image=qr_img_tk).grid(row=1, column=0, padx=10, pady=10)
        tb.Label(qr_window, text=url, font=("Segoe UI", 12)).grid(row=2, column=0, padx=10, pady=10)
        # Keep a reference to the image.
        qr_window.qr_img_tk = qr_img_tk

