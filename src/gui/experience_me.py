import os
import threading
import ttkbootstrap as tb
import cv2
import numpy as np
import qrcode
import tkinter as tk
import matplotlib.pyplot as plt
import math
from ttkbootstrap.constants import *
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
from calibration_db import load_calibrations
from capture_server import start_capture_server_in_thread, get_local_ip
from camera_calibration import CameraCalibrator 
from real_coordinates import CoordinateTransformer

# Compute the project root (three levels up)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHOTOS_FOLDER = os.path.join(BASE_DIR, "data\\images")


class Experience(tb.Frame):
    def __init__(self, master):
        """
        Initialize the Experience frame with UI elements for camera options,
        captured media, extraction, and motion analysis.
        """
        super().__init__(master, padding=10)
        self.grid(sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)  # The center area expands
        self.bind("<Visibility>", self.on_visibility)

        # Top frame for Camera Options and QR Code
        top_frame = tb.Frame(self)
        top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=1)

        # Left column: Camera Options and Action Buttons
        cam_options_frame = tb.Labelframe(top_frame, text="Camera Options", bootstyle="info")
        cam_options_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        cam_options_frame.columnconfigure(1, weight=1)

        tb.Label(cam_options_frame, text="Camera Name:", font=("Segoe UI", 12))\
            .grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.camera_choice = tb.Combobox(cam_options_frame, bootstyle="info")
        self.camera_choice.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.update_camera_choices()

        tb.Label(cam_options_frame, text="Square Size (mm):", font=("Segoe UI", 12))\
            .grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.square_size_entry = tb.Entry(cam_options_frame, font=("Segoe UI", 12))
        self.square_size_entry.insert(0, "30.0")
        self.square_size_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        tb.Label(cam_options_frame, text="Pattern Size (e.g. 4,4):", font=("Segoe UI", 12))\
            .grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.pattern_size_entry = tb.Entry(cam_options_frame, font=("Segoe UI", 12))
        self.pattern_size_entry.insert(0, "4,4")
        self.pattern_size_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

        # Action Buttons for capturing media
        action_buttons_frame = tb.Frame(cam_options_frame)
        action_buttons_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        action_buttons_frame.columnconfigure(0, weight=1)
        action_buttons_frame.columnconfigure(1, weight=1)
        action_buttons_frame.columnconfigure(2, weight=1)
        tb.Button(action_buttons_frame, text="Capture Images from Phone", 
                  command=self.capture_video_from_phone, bootstyle="primary")\
            .grid(row=0, column=0, sticky="ew", padx=3, pady=3)
        tb.Button(action_buttons_frame, text="Capture Video from Phone", 
                  command=self.capture_video_from_phone, bootstyle="success")\
            .grid(row=0, column=1, sticky="ew", padx=3, pady=3)
        tb.Button(action_buttons_frame, text="Refresh Captured Images", 
                  command=self.load_captured_images, bootstyle="warning")\
            .grid(row=0, column=2, sticky="ew", padx=3, pady=3)

        # Right column: QR Code display
        self.qr_frame = tb.Labelframe(top_frame, text="QR Code", bootstyle="info")
        self.qr_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        tb.Label(self.qr_frame, text="Le QR Code s'affichera ici", font=("Segoe UI", 12)).pack(padx=10, pady=10)

        # Center frame with two side-by-side blocks
        center_frame = tb.Frame(self)
        center_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        center_frame.columnconfigure(0, weight=1)
        center_frame.columnconfigure(1, weight=5)
        center_frame.rowconfigure(0, weight=1)

        # Left block: Captured Media with scrollable thumbnails
        captured_frame = tb.Labelframe(center_frame, text="Captured Media (Cliquez pour sélectionner/désélectionner)", bootstyle="info")
        captured_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        captured_frame.columnconfigure(0, weight=1)
        captured_frame.rowconfigure(0, weight=1)
        captured_frame.configure(width=200)
        captured_frame.grid_propagate(False)

        self.images_canvas = tb.Canvas(captured_frame, background="#ffffff", bd=0, highlightthickness=0)
        self.images_canvas.grid(row=0, column=0, sticky="nsew")
        self.images_canvas.config(height=300)
        self.vscroll = tb.Scrollbar(captured_frame, orient="vertical", command=self.images_canvas.yview)
        self.vscroll.grid(row=0, column=1, sticky="ns", padx=5, pady=5)
        self.images_canvas.configure(yscrollcommand=self.vscroll.set)
        self.images_canvas.bind("<Configure>", self.on_canvas_configure)
        self.images_frame = tb.Frame(self.images_canvas)
        self.images_canvas.create_window((0, 0), window=self.images_frame, anchor="nw")
        self.images_frame.bind("<Configure>", lambda e: self.images_canvas.configure(scrollregion=self.images_canvas.bbox("all")))

        upload_button = tb.Button(captured_frame, text="Upload Selected Video", command=self.upload_video, bootstyle="warning")
        upload_button.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        extraire_button = tb.Button(captured_frame, text="Extraire la matrice d'homographie", 
                                    command=self.extract_homography_matrix, bootstyle="info")
        extraire_button.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        # Right block: Extraction & Traitement
        self.additional_block = tb.Labelframe(center_frame, text="Extraction & Traitement", bootstyle="info")
        self.additional_block.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.additional_block.columnconfigure(0, weight=1)
        self.additional_block.columnconfigure(1, weight=1)
        self.additional_block.rowconfigure(3, weight=1)

        # Row 0: Slider controls for start and end frames
        start_controls = tb.Frame(self.additional_block)
        start_controls.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        start_label = tb.Label(start_controls, text="Frame de début :", font=("Segoe UI", 10))
        start_label.pack(side="left", padx=(0, 5))
        self.start_frame_slider = tb.Scale(start_controls, from_=0, to=100, orient="horizontal",
                                           bootstyle="info", length=150, command=self.update_start_frame_preview)
        self.start_frame_slider.pack(side="left")

        end_controls = tb.Frame(self.additional_block)
        end_controls.grid(row=0, column=1, sticky="w", padx=10, pady=5)
        end_label = tb.Label(end_controls, text="Frame de fin :", font=("Segoe UI", 10))
        end_label.pack(side="left", padx=(0, 5))
        self.end_frame_slider = tb.Scale(end_controls, from_=0, to=100, orient="horizontal",
                                         bootstyle="info", length=150, command=self.update_end_frame_preview)
        self.end_frame_slider.pack(side="left")

        # Row 1: Preview of the selected frames
        self.start_preview_label = tb.Label(self.additional_block, relief="sunken")
        self.start_preview_label.grid(row=1, column=0, padx=10, pady=5)
        self.end_preview_label = tb.Label(self.additional_block, relief="sunken")
        self.end_preview_label.grid(row=1, column=1, padx=10, pady=5)
        
        # Row 2: Video mode selection (Normal vs Slow Motion at 240 fps)
        video_mode_frame = tb.Frame(self.additional_block)
        video_mode_frame.grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=5)
        tb.Label(video_mode_frame, text="Video Mode:", font=("Segoe UI", 10)).pack(side="left", padx=(0,5))
        self.video_mode = tk.StringVar(value="normal")
        tb.Radiobutton(video_mode_frame, text="Normal", variable=self.video_mode, value="normal", bootstyle="info")\
            .pack(side="left", padx=5)
        tb.Radiobutton(video_mode_frame, text="Slow Motion (240 fps)", variable=self.video_mode, value="slow", bootstyle="info")\
            .pack(side="left", padx=5)
        
        # Row 3: Action buttons for extraction and analysis
        extract_button = tb.Button(self.additional_block, text="Extraire",
                                   command=self.threaded_extract_frames, bootstyle="success")
        extract_button.grid(row=3, column=1, sticky="se", padx=10, pady=10)
        analyse_button = tb.Button(self.additional_block, text="Analyse Motion",
                                   command=self.analyze_motion_instantaneous, bootstyle="success")
        analyse_button.grid(row=3, column=2, sticky="se", padx=10, pady=10)
        preview_button = tb.Button(self.additional_block, text="Preview Detection (External)",
                                   command=self.preview_detection_external, bootstyle="secondary")
        preview_button.grid(row=3, column=0, sticky="se", padx=10, pady=10)

        # Initialize state variables
        self.calib_files = [] 
        self.thumbnail_images = {} 
        self.current_camera_folder = None
        self.cap = None
        self.selected_video_path = None  # currently selected video path
        self.uploaded_video_path = None  # stored video path for next treatment
        self.selected_video_button = None
        self.coordinate_transformer = None

    def on_visibility(self, event):
        """Refresh camera choices when the widget becomes visible."""
        self.update_camera_choices()

    def update_camera_choices(self):
        """Update the camera choice combobox with available calibrations."""
        calibrations = load_calibrations()
        cam_names = list(calibrations.keys())
        self.camera_choice['values'] = cam_names

    def ensure_camera_folder(self, camera_name):
        """
        Ensure that the folder for the selected camera exists.
        Returns the camera folder path.
        """
        if not camera_name:
            return None
        if not os.path.exists(PHOTOS_FOLDER):
            os.makedirs(PHOTOS_FOLDER)
        camera_folder = os.path.join(PHOTOS_FOLDER, camera_name)
        if not os.path.exists(camera_folder):
            os.makedirs(camera_folder)
        self.current_camera_folder = camera_folder
        return camera_folder

    def toggle_image_selection(self, full_path, button):
        """Toggle the selection state for an image thumbnail."""
        if full_path in self.calib_files:
            self.calib_files.remove(full_path)
            button.configure(bootstyle="link")
        else:
            self.calib_files.append(full_path)
            button.configure(bootstyle="success")

    def toggle_video_selection(self, full_path, button):
        """Toggle the selection state for a video thumbnail."""
        if self.selected_video_path == full_path:
            self.selected_video_path = None
            button.configure(bootstyle="link")
        else:
            self.selected_video_path = full_path
            button.configure(bootstyle="success")
            if self.selected_video_button and self.selected_video_button is not button:
                self.selected_video_button.configure(bootstyle="link")
            self.selected_video_button = button

    def capture_video_from_phone(self):
        """
        Start the capture server for the selected camera and display the QR code
        for the phone to connect.
        """
        selected_camera = self.camera_choice.get().strip()
        if not selected_camera:
            messagebox.showerror("Error", "Veuillez renseigner un nom de caméra.")
            return
        camera_folder = self.ensure_camera_folder(selected_camera)
        start_capture_server_in_thread(target_folder=camera_folder)
        local_ip = get_local_ip()
        url = f"https://{local_ip}:5000"
        qr = qrcode.QRCode(version=1, box_size=4, border=4)
        qr.add_data(url)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_img_tk = ImageTk.PhotoImage(qr_img)
        for widget in self.qr_frame.winfo_children():
            widget.destroy()
        tb.Label(self.qr_frame, text="Scannez ce QR code avec votre téléphone :", font=("Segoe UI", 8, "bold"))\
            .pack(padx=10, pady=2)
        tb.Label(self.qr_frame, image=qr_img_tk).pack(padx=10, pady=5)
        tb.Label(self.qr_frame, text=url, font=("Segoe UI", 12))\
            .pack(padx=10, pady=2)
        self.qr_frame.qr_img_tk = qr_img_tk

    def on_canvas_configure(self, event):
        """
        Reconfigure the grid of thumbnail buttons when the canvas size changes.
        """
        canvas_width = event.width
        thumb_size = 110
        num_columns = max(1, canvas_width // thumb_size)
        for idx, widget in enumerate(self.images_frame.winfo_children()):
            row = idx // num_columns
            col = idx % num_columns
            widget.grid_configure(row=row, column=col)

    def load_captured_images(self):
        """
        Scan the current camera folder and load images and video thumbnails into the UI.
        """
        for widget in self.images_frame.winfo_children():
            widget.destroy()
        camera_name = self.camera_choice.get().strip()
        if camera_name:
            self.ensure_camera_folder(camera_name)
        if os.path.exists(self.current_camera_folder):
            files = sorted(os.listdir(self.current_camera_folder))
            image_extensions = ('.jpg', '.jpeg', '.png')
            video_extensions = ('.mp4', '.avi', '.mov')
            combined_files = [f for f in files if f.lower().endswith(image_extensions + video_extensions)]
            for idx, filename in enumerate(combined_files):
                full_path = os.path.join(self.current_camera_folder, filename)
                if filename.lower().endswith(image_extensions):
                    try:
                        img = Image.open(full_path)
                        img.thumbnail((100, 100))
                        photo = ImageTk.PhotoImage(img)
                        self.thumbnail_images[full_path] = photo
                        style = "success" if full_path in self.calib_files else "link"
                        btn = tb.Button(self.images_frame, image=photo, bootstyle=style)
                        btn.config(command=lambda fp=full_path, b=btn: self.toggle_image_selection(fp, b))
                        btn.grid(row=0, column=idx, padx=5, pady=5)
                    except Exception as e:
                        print("Erreur lors du chargement de l'image :", full_path, e)
                elif filename.lower().endswith(video_extensions):
                    if not hasattr(self, 'video_icon'):
                        video_icon = Image.new('RGB', (100, 100), color='gray')
                        draw = ImageDraw.Draw(video_icon)
                        draw.polygon([(35, 25), (35, 75), (75, 50)], fill='white')
                        self.video_icon = ImageTk.PhotoImage(video_icon)
                    self.thumbnail_images[full_path] = self.video_icon
                    btn = tb.Button(self.images_frame, image=self.video_icon, bootstyle="link")
                    btn.config(command=lambda fp=full_path, b=btn: self.toggle_video_selection(fp, b))
                    btn.grid(row=0, column=idx, padx=5, pady=5)
        else:
            messagebox.showerror("Error", f"Photos folder not found at {PHOTOS_FOLDER}.")

    def upload_video(self):
        """
        Set the currently selected video as the uploaded video and update frame sliders.
        """
        if self.selected_video_path:
            self.uploaded_video_path = self.selected_video_path
            self.cap = cv2.VideoCapture(self.uploaded_video_path)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.start_frame_slider.config(to=total_frames - 1)
            self.end_frame_slider.config(to=total_frames - 1)
            messagebox.showinfo("Upload", f"Video selected for upload:\n{self.uploaded_video_path}")
        else:
            messagebox.showwarning("Upload", "Aucune vidéo sélectionnée!")

    def extract_homography_matrix(self):
        """
        Extract the homography matrix from the first selected calibration image.
        """
        if not self.calib_files:
            messagebox.showwarning("Erreur", "Aucune image sélectionnée dans Captured Images.")
            return
        image_path = self.calib_files[0]
        img = cv2.imread(image_path)
        if img is None:
            messagebox.showerror("Erreur", f"Impossible de charger l'image : {image_path}")
            return
        selected_camera = self.camera_choice.get().strip()
        if not selected_camera:
            messagebox.showerror("Erreur", "Aucune caméra sélectionnée.")
            return
        calibrations = load_calibrations()
        if selected_camera not in calibrations:
            messagebox.showerror("Erreur", f"Calibration introuvable pour la caméra '{selected_camera}'.")
            return
        calibration = calibrations[selected_camera]
        camera_matrix = np.array(calibration["camera_matrix"])
        dist_coefs = np.array(calibration["dist_coefs"])
        pattern_size = calibration["pattern_size"]
        transformer = CoordinateTransformer(camera_matrix, dist_coefs, pattern_size)
        if transformer.compute_homography(img):
            self.coordinate_transformer = transformer
            messagebox.showinfo("Homographie",
                                f"Homographie calculée avec succès.\n\nMatrice H :\n{transformer.H_inv}")
        else:
            messagebox.showerror("Erreur",
                                "Échec du calcul de la matrice d'homographie.\n"
                                "Vérifiez que l'image contient bien le motif (échiquier) attendu.")

    def update_frame_preview(self, label, frame_number):
        """
        Update the provided label with a preview image from the video at the given frame number.
        """
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        scale_factor = 0.2
        video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        image = image.resize((int(video_width * scale_factor), int(video_height * scale_factor)))
        image_tk = ImageTk.PhotoImage(image)
        label.config(image=image_tk)
        label.image = image_tk

    def update_start_frame_preview(self, value):
        """Update the start frame preview based on the slider value."""
        frame_number = int(float(value))
        self.update_frame_preview(self.start_preview_label, frame_number)

    def update_end_frame_preview(self, value):
        """Update the end frame preview based on the slider value."""
        frame_number = int(float(value))
        self.update_frame_preview(self.end_preview_label, frame_number)

    def calculate_fps(self):
        """Calculate and return the FPS of the uploaded video."""
        cap = cv2.VideoCapture(self.uploaded_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

    def detect_object_center(self, frame):
        """
        Detect the moving object's center in the provided frame using background subtraction.
        Returns a tuple (cx, cy) or None if not found.
        """
        if not hasattr(self, 'bg_subtractor'):
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        fg_mask = self.bg_subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return (cx, cy)
        return None

    def preview_detection_external(self):
        """
        Open a new window to preview object detection on the uploaded video.
        """
        if not self.uploaded_video_path:
            messagebox.showwarning("Erreur", "Aucune vidéo n'a été sélectionnée!")
            return
        self.preview_window = tk.Toplevel(self)
        self.preview_window.title("Detection Preview")
        self.preview_window.geometry("400x300")
        self.preview_label = tb.Label(self.preview_window)
        self.preview_label.pack(padx=10, pady=10)
        self.cap_preview = cv2.VideoCapture(self.uploaded_video_path)
        self.update_preview()

    def update_preview(self):
        """
        Continuously update the preview window with detection results.
        """
        ret, frame = self.cap_preview.read()
        if ret:
            center = self.detect_object_center(frame)
            if center is not None:
                cv2.circle(frame, center, 10, (0, 255, 0), 2)
            scale_factor = 0.3
            frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(img)
            self.preview_label.config(image=img_tk)
            self.preview_label.image = img_tk
            self.preview_window.after(30, self.update_preview)
        else:
            self.cap_preview.release()
            self.preview_window.destroy()

    def detect_movement_between_frames(self, start_frame, end_frame):
        """
        Detect movement between two specified frames and return
        the start and end centers, displacement, and Euclidean distance.
        """
        cap = cv2.VideoCapture(self.uploaded_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame < 0 or end_frame >= total_frames or start_frame > end_frame:
            messagebox.showerror("Erreur", "Valeurs de frames invalides!")
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame_start = cap.read()
        if not ret:
            messagebox.showerror("Erreur", "Impossible de lire la frame de début")
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
        ret, frame_end = cap.read()
        if not ret:
            messagebox.showerror("Erreur", "Impossible de lire la frame de fin")
            cap.release()
            return None
        center_start = self.detect_object_center(frame_start)
        center_end = self.detect_object_center(frame_end)
        cap.release()
        if center_start is None or center_end is None:
            messagebox.showerror("Erreur", "L'objet n'a pas été détecté dans l'une des frames")
            return None
        dx = center_end[0] - center_start[0]
        dy = center_end[1] - center_start[1]
        displacement = (dx, dy)
        distance = math.sqrt(dx * dx + dy * dy)
        return center_start, center_end, displacement, distance

    def extract_frames(self):
        """
        Extract frames from the uploaded video between the selected start and end frames.
        A progress bar is displayed during extraction.
        """
        if not self.uploaded_video_path:
            messagebox.showwarning("Extraction", "Aucune vidéo n'a été sélectionnée!")
            return
        start_frame = int(self.start_frame_slider.get())
        end_frame = int(self.end_frame_slider.get())
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.uploaded_video_path)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame < 0 or end_frame >= total_frames or start_frame > end_frame:
            messagebox.showerror("Erreur", "Valeurs de frames invalides!")
            return

        output_folder = os.path.join(os.path.dirname(self.uploaded_video_path), "extracted_frames")
        os.makedirs(output_folder, exist_ok=True)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame

        # Create a progress bar widget
        progress = tb.Progressbar(self, mode='determinate', bootstyle="info")
        progress.grid(row=4, column=0, columnspan=2, padx=10, pady=10)
        progress['maximum'] = end_frame - start_frame + 1

        try:
            while current_frame <= end_frame:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame_filename = os.path.join(output_folder, f"frame_{current_frame}.jpg")
                cv2.imwrite(frame_filename, frame)
                current_frame += 1
                progress.step(1)
                self.update()  # Refresh the UI
        finally:
            progress.destroy()
        messagebox.showinfo("Extraction Terminée",
                            f"Extraction terminée.\nFrames extraites: {current_frame - start_frame}\nDossier: {output_folder}")

    def threaded_extract_frames(self):
        """Run extract_frames in a separate thread to keep the UI responsive."""
        threading.Thread(target=self.extract_frames, daemon=True).start()

    def analyze_motion_instantaneous(self):
        """
        Analyze the instantaneous motion (speed and acceleration) of the object
        between the selected frames using the homography matrix.
        
        The time interval (dt) is computed based on the video mode:
          - For Normal mode: dt = 1.0 / (video FPS)
          - For Slow Motion mode: dt = 1.0 / 240
        """
        if not self.coordinate_transformer:
            messagebox.showerror("Erreur", "La matrice d'homographie n'a pas été calculée.")
            return

        start_frame = int(self.start_frame_slider.get())
        end_frame = int(self.end_frame_slider.get())
        cap = cv2.VideoCapture(self.uploaded_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame < 0 or end_frame >= total_frames or start_frame >= end_frame:
            messagebox.showerror("Erreur", "Valeurs de frames invalides!")
            cap.release()
            return

        # Set dt based on video mode selection
        if self.video_mode.get() == "slow":
            dt = 1.0 / 240.0
        else:
            dt = 1.0 / fps

        try:
            square_size = float(self.square_size_entry.get().strip())
        except ValueError:
            messagebox.showerror("Erreur", "Square Size invalide!")
            cap.release()
            return

        centers_world = []
        frame_numbers = []
        for frame_no in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret:
                centers_world.append(None)
            else:
                # Undistort the frame using calibration parameters
                undistorted_frame = cv2.undistort(
                    frame,
                    self.coordinate_transformer.camera_matrix,
                    self.coordinate_transformer.dist_coefs,
                    None,
                    self.coordinate_transformer.camera_matrix
                )
                center = self.detect_object_center(undistorted_frame)
                if center is None:
                    centers_world.append(None)
                else:
                    center_array = np.array([[center]], dtype=np.float32)
                    world_pt = cv2.perspectiveTransform(center_array, self.coordinate_transformer.H_inv)
                    # Convert from mm to cm (mm/10 = cm)
                    world_pt = world_pt[0, 0] * square_size / 10.0
                    centers_world.append(world_pt)
            frame_numbers.append(frame_no)
        cap.release()

        instantaneous_speeds = []
        for i in range(1, len(centers_world)):
            if centers_world[i] is None or centers_world[i - 1] is None:
                instantaneous_speeds.append(None)
            else:
                dx = centers_world[i][0] - centers_world[i - 1][0]
                dy = centers_world[i][1] - centers_world[i - 1][1]
                dist = math.sqrt(dx * dx + dy * dy)  # displacement in cm
                speed_cm_s = dist / dt  # cm/s
                speed_m_s = speed_cm_s / 100.0  # m/s
                instantaneous_speeds.append(speed_m_s)

        # Compute acceleration using finite differences
        instantaneous_acc = []
        for i in range(len(instantaneous_speeds)):
            if instantaneous_speeds[i] is None:
                instantaneous_acc.append(None)
            elif i == 0:
                if instantaneous_speeds[i + 1] is not None:
                    a = (instantaneous_speeds[i + 1] - instantaneous_speeds[i]) / dt
                    instantaneous_acc.append(a)
                else:
                    instantaneous_acc.append(None)
            elif i == len(instantaneous_speeds) - 1:
                if instantaneous_speeds[i - 1] is not None:
                    a = (instantaneous_speeds[i] - instantaneous_speeds[i - 1]) / dt
                    instantaneous_acc.append(a)
                else:
                    instantaneous_acc.append(None)
            else:
                if instantaneous_speeds[i + 1] is not None and instantaneous_speeds[i - 1] is not None:
                    a = (instantaneous_speeds[i + 1] - instantaneous_speeds[i - 1]) / (2 * dt)
                    instantaneous_acc.append(a)
                else:
                    instantaneous_acc.append(None)

        valid_speeds = [s for s in instantaneous_speeds if s is not None]
        valid_acc = [a for a in instantaneous_acc if a is not None]
        avg_speed = np.mean(valid_speeds) if valid_speeds else 0
        avg_acc = np.mean(valid_acc) if valid_acc else 0

        result_str = (
            f"Analyse Instantanée de {start_frame} à {end_frame}:\n"
            f"  Vitesse moyenne: {avg_speed:.2f} m/s\n"
            f"  Accélération moyenne: {avg_acc:.2f} m/s²"
        )
        messagebox.showinfo("Instantaneous Motion Analysis", result_str)

        plt.figure("Instantaneous Speed")
        plt.plot(frame_numbers[1:], instantaneous_speeds, marker='o', label="Speed (m/s)")
        plt.xlabel("Frame Number")
        plt.ylabel("Speed (m/s)")
        plt.title("Instantaneous Speed vs Frame")
        plt.legend()
        plt.show()

        plt.figure("Instantaneous Acceleration")
        plt.plot(frame_numbers, instantaneous_acc, marker='x', color='red', label="Acceleration (m/s²)")
        plt.xlabel("Frame Number")
        plt.ylabel("Acceleration (m/s²)")
        plt.title("Instantaneous Acceleration vs Frame")
        plt.legend()
        plt.show()
