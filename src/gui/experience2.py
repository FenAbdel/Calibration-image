import os
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

# Calcul du répertoire racine du projet (trois niveaux au-dessus)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHOTOS_FOLDER = os.path.join(BASE_DIR, "data\\images")

class Experience(tb.Frame):
    def __init__(self, master):
        super().__init__(master, padding=10)
        self.grid(sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.bind("<Visibility>", self.on_visibility)
        
        # Initialisation du filtre de Kalman à None
        self.kalman = None

        # --- Zone supérieure ---
        top_frame = tb.Frame(self)
        top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=1)
        
        # Options de la caméra (colonne de gauche)
        cam_options_frame = tb.Labelframe(top_frame, text="Options de la caméra", bootstyle="info")
        cam_options_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        cam_options_frame.columnconfigure(1, weight=1)
        
        tb.Label(cam_options_frame, text="Nom de la caméra :", font=("Segoe UI", 12))\
            .grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.camera_choice = tb.Combobox(cam_options_frame, bootstyle="info")
        self.camera_choice.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.update_camera_choices()

        tb.Label(cam_options_frame, text="Taille du carré (mm) :", font=("Segoe UI", 12))\
            .grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.square_size_entry = tb.Entry(cam_options_frame, font=("Segoe UI", 12))
        self.square_size_entry.insert(0, "30.0")
        self.square_size_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        tb.Label(cam_options_frame, text="Taille du motif (ex. 4,4) :", font=("Segoe UI", 12))\
            .grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.pattern_size_entry = tb.Entry(cam_options_frame, font=("Segoe UI", 12))
        self.pattern_size_entry.insert(0, "4,4")
        self.pattern_size_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        
        # Nouvelle option : Choix de la forme de l'objet
        tb.Label(cam_options_frame, text="Forme de l'objet :", font=("Segoe UI", 12))\
            .grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.shape_choice = tb.Combobox(cam_options_frame, bootstyle="info", 
                                        values=["Blob", "Cercle", "Contour"])
        self.shape_choice.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
        self.shape_choice.set("Blob")
        
        # Boutons d'action (placés en dessous)
        action_buttons_frame = tb.Frame(cam_options_frame)
        action_buttons_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        action_buttons_frame.columnconfigure(0, weight=1)
        action_buttons_frame.columnconfigure(1, weight=1)
        action_buttons_frame.columnconfigure(2, weight=1)
        tb.Button(action_buttons_frame, text="Capturer des images depuis le téléphone", 
                  command=self.capture_video_from_phone, bootstyle="primary")\
            .grid(row=0, column=0, sticky="ew", padx=3, pady=3)
        tb.Button(action_buttons_frame, text="Capturer une vidéo depuis le téléphone", 
                  command=self.capture_video_from_phone, bootstyle="success")\
            .grid(row=0, column=1, sticky="ew", padx=3, pady=3)
        tb.Button(action_buttons_frame, text="Rafraîchir les médias capturés", 
                  command=self.load_captured_images, bootstyle="warning")\
            .grid(row=0, column=2, sticky="ew", padx=3, pady=3)
            
        self.qr_frame = tb.Labelframe(top_frame, text="QR Code", bootstyle="info")
        self.qr_frame.grid(row=0, column=1, rowspan=1, sticky="nsew", padx=5, pady=5)
        tb.Label(self.qr_frame, text="Le QR Code s'affichera ici", font=("Segoe UI", 12))\
            .pack(padx=10, pady=10)
        
        # --- Zone centrale (Deux colonnes) ---
        center_frame = tb.Frame(self)
        center_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        center_frame.columnconfigure(0, weight=1)
        center_frame.columnconfigure(1, weight=5)
        center_frame.rowconfigure(0, weight=1)

        # Médias capturés (colonne de gauche)
        captured_frame = tb.Labelframe(center_frame, text="Médias capturés (Cliquez pour sélectionner/désélectionner)", bootstyle="info")
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
        
        tb.Button(captured_frame, text="Téléverser la vidéo sélectionnée", command=self.upload_video, bootstyle="warning")\
            .grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        tb.Button(captured_frame, text="Extraire la matrice d'homographie", command=self.extract_homography_matrix, bootstyle="info")\
            .grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        # Bloc Extraction et Traitement (colonne de droite)
        self.additional_block = tb.Labelframe(center_frame, text="Extraction et Traitement", bootstyle="info")
        self.additional_block.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.additional_block.columnconfigure(0, weight=1)
        self.additional_block.columnconfigure(1, weight=1)
        self.additional_block.rowconfigure(2, weight=1)

        start_controls = tb.Frame(self.additional_block)
        start_controls.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        tb.Label(start_controls, text="Cadre de début :", font=("Segoe UI", 10))\
            .pack(side="left", padx=(0, 5))
        self.start_frame_slider = tb.Scale(
            start_controls,
            from_=0,
            to=100,
            orient="horizontal",
            bootstyle="info",
            length=150,
            command=self.update_start_frame_preview
        )
        self.start_frame_slider.pack(side="left")

        end_controls = tb.Frame(self.additional_block)
        end_controls.grid(row=0, column=1, sticky="w", padx=10, pady=5)
        tb.Label(end_controls, text="Cadre de fin :", font=("Segoe UI", 10))\
            .pack(side="left", padx=(0, 5))
        self.end_frame_slider = tb.Scale(
            end_controls,
            from_=0,
            to=100,
            orient="horizontal",
            bootstyle="info",
            length=150,
            command=self.update_end_frame_preview
        )
        self.end_frame_slider.pack(side="left")

        self.start_preview_label = tb.Label(self.additional_block, relief="sunken")
        self.start_preview_label.grid(row=1, column=0, padx=10, pady=5)
        self.end_preview_label = tb.Label(self.additional_block, relief="sunken")
        self.end_preview_label.grid(row=1, column=1, padx=10, pady=5)
        
        # La ligne de bouton "Extraire" a été supprimée.
        tb.Button(self.additional_block, text="Analyser le mouvement", command=self.analyze_motion_instantaneous, bootstyle="success")\
            .grid(row=3, column=2, sticky="se", padx=10, pady=10)
        tb.Button(self.additional_block, text="Prévisualiser la détection (externe)", command=self.preview_detection_external, bootstyle="secondary")\
            .grid(row=3, column=0, sticky="se", padx=10, pady=10)
        
        # Variables d'état internes
        self.calib_files = [] 
        self.thumbnail_images = {} 
        self.current_camera_folder = None
        self.cap = None
        self.selected_video_path = None  # chemin de la vidéo sélectionnée
        self.uploaded_video_path = None  # chemin stocké pour le traitement
        self.selected_video_button = None
        
        self.frame1_selected = None  # Cadre de début
        self.frame2_selected = None  # Cadre intermédiaire
        self.frame3_selected = None  # Cadre de fin

    def on_visibility(self, event):
        self.update_camera_choices()
    
    def update_camera_choices(self):
        calibrations = load_calibrations()
        cam_names = list(calibrations.keys())
        self.camera_choice['values'] = cam_names

    def ensure_camera_folder(self, camera_name):
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
        if full_path in self.calib_files:
            self.calib_files.remove(full_path)
            button.configure(bootstyle="link")
        else:
            self.calib_files.append(full_path)
            button.configure(bootstyle="success")

    def toggle_video_selection(self, full_path, button):
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
        selected_camera = self.camera_choice.get()
        if not selected_camera:
            messagebox.showerror("Erreur", "Veuillez renseigner un nom de caméra.")
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
        canvas_width = event.width
        thumb_size = 110
        num_columns = max(1, canvas_width // thumb_size)
        for idx, widget in enumerate(self.images_frame.winfo_children()):
            row = idx // num_columns
            col = idx % num_columns
            widget.grid_configure(row=row, column=col)
     
    def load_captured_images(self):
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
            messagebox.showerror("Erreur", f"Dossier d'images non trouvé à {PHOTOS_FOLDER}.")
    
    def upload_video(self):
        if self.selected_video_path:
            self.uploaded_video_path = self.selected_video_path
            self.cap = cv2.VideoCapture(self.uploaded_video_path)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.start_frame_slider.config(to=total_frames - 1)
            self.end_frame_slider.config(to=total_frames - 1)
            messagebox.showinfo("Téléversement", f"Vidéo sélectionnée pour téléversement :\n{self.uploaded_video_path}")
        else:
            messagebox.showwarning("Téléversement", "Aucune vidéo sélectionnée !")

    def extract_homography_matrix(self):
        if not self.calib_files:
            messagebox.showwarning("Erreur", "Aucune image sélectionnée dans les médias capturés.")
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

    def extract_frames(self):
        if not self.uploaded_video_path:
            messagebox.showwarning("Extraction", "Aucune vidéo n'a été sélectionnée !")
            return
        start_frame = int(self.start_frame_slider.get())
        end_frame = int(self.end_frame_slider.get())
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.uploaded_video_path)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame < 0 or end_frame >= total_frames or start_frame > end_frame:
            messagebox.showerror("Erreur", "Valeurs de cadres invalides !")
            return
        output_folder = os.path.join(os.path.dirname(self.uploaded_video_path), "frames_extraites")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_filename = os.path.join(output_folder, f"frame_{current_frame}.jpg")
            cv2.imwrite(frame_filename, frame)
            current_frame += 1
        messagebox.showinfo("Extraction Terminée",
                            f"Extraction terminée.\nCadres extraits : {current_frame - start_frame}\nDossier : {output_folder}")
    
    def update_start_frame_preview(self, value):
        frame_number = int(float(value))
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale_factor = 0.2
        display_width = int(video_width * scale_factor)
        display_height = int(video_height * scale_factor)
        image = image.resize((display_width, display_height))
        image_tk = ImageTk.PhotoImage(image)
        self.start_preview_label.config(image=image_tk)
        self.start_preview_label.image = image_tk

    def update_end_frame_preview(self, value):
        frame_number = int(float(value))
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale_factor = 0.2
        display_width = int(video_width * scale_factor)
        display_height = int(video_height * scale_factor)
        image = image.resize((display_width, display_height))
        image_tk = ImageTk.PhotoImage(image)
        self.end_preview_label.config(image=image_tk)
        self.end_preview_label.image = image_tk
        
    def calculate_fps(self):
        cap = cv2.VideoCapture(self.uploaded_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

    # --- Méthodes améliorées de détection du centre de l'objet ---
    def detect_object_center_blob(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 50      
        params.maxArea = 5000    
        params.filterByCircularity = False  
        params.filterByConvexity = False
        params.filterByInertia = False
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(blurred)
        if keypoints:
            best_keypoint = max(keypoints, key=lambda kp: kp.size)
            cx, cy = int(best_keypoint.pt[0]), int(best_keypoint.pt[1])
            return (cx, cy)
        return None

    def detect_object_center_circle(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                   param1=50, param2=30, minRadius=10, maxRadius=100)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            circle = circles[0, 0]
            cx, cy, r = circle
            return (cx, cy)
        return None

    def detect_object_center_contour(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
        return None

    def initialize_kalman(self, center):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        self.kalman.statePost = np.array([[np.float32(center[0])],
                                          [np.float32(center[1])],
                                          [0],
                                          [0]], np.float32)

    def detect_object_center(self, frame):
        shape_type = self.shape_choice.get() if hasattr(self, 'shape_choice') else "Blob"
        if shape_type == "Blob":
            measurement_center = self.detect_object_center_blob(frame)
        elif shape_type == "Cercle":
            measurement_center = self.detect_object_center_circle(frame)
        elif shape_type == "Contour":
            measurement_center = self.detect_object_center_contour(frame)
        else:
            measurement_center = self.detect_object_center_blob(frame)
        
        if measurement_center is None:
            return None
        
        if not hasattr(self, 'kalman') or self.kalman is None:
            self.initialize_kalman(measurement_center)
            return measurement_center
        else:
            measurement = np.array([[np.float32(measurement_center[0])],
                                    [np.float32(measurement_center[1])]])
            self.kalman.correct(measurement)
            predicted = self.kalman.predict()
            predicted_center = (int(predicted[0]), int(predicted[1]))
            return predicted_center

    def preview_detection_external(self):
        if not self.uploaded_video_path:
            messagebox.showwarning("Erreur", "Aucune vidéo n'a été sélectionnée !")
            return
        self.preview_window = tk.Toplevel(self)
        self.preview_window.title("Prévisualisation de la détection")
        self.preview_window.geometry("400x300")
        self.preview_label = tb.Label(self.preview_window)
        self.preview_label.pack(padx=10, pady=10)
        self.cap_preview = cv2.VideoCapture(self.uploaded_video_path)
        self.update_preview()

    def update_preview(self):
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
        cap = cv2.VideoCapture(self.uploaded_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame < 0 or end_frame >= total_frames or start_frame > end_frame:
            messagebox.showerror("Erreur", "Valeurs de cadres invalides !")
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame_start = cap.read()
        if not ret:
            messagebox.showerror("Erreur", "Impossible de lire le cadre de début")
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
        ret, frame_end = cap.read()
        if not ret:
            messagebox.showerror("Erreur", "Impossible de lire le cadre de fin")
            cap.release()
            return None
        center_start = self.detect_object_center(frame_start)
        center_end = self.detect_object_center(frame_end)
        cap.release()
        if center_start is None or center_end is None:
            messagebox.showerror("Erreur", "L'objet n'a pas été détecté dans l'un des cadres")
            return None
        dx = center_end[0] - center_start[0]
        dy = center_end[1] - center_start[1]
        displacement = (dx, dy)
        distance = math.sqrt(dx * dx + dy * dy)
        return center_start, center_end, displacement, distance

    def analyze_motion_instantaneous(self):
        if not hasattr(self, 'coordinate_transformer'):
            messagebox.showerror("Erreur", "La matrice d'homographie n'a pas été calculée.")
            return

        start_frame = int(self.start_frame_slider.get())
        end_frame = int(self.end_frame_slider.get())
        cap = cv2.VideoCapture(self.uploaded_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame < 0 or end_frame >= total_frames or start_frame >= end_frame:
            messagebox.showerror("Erreur", "Valeurs de cadres invalides !")
            cap.release()
            return

        try:
            square_size = float(self.square_size_entry.get().strip())
        except ValueError:
            messagebox.showerror("Erreur", "Taille du carré invalide !")
            cap.release()
            return

        dt = 1.0 / fps
        centers_world = []
        frame_numbers = []
        for frame_no in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret:
                centers_world.append(None)
            else:
                undistorted_frame = cv2.undistort(frame, 
                                                  self.coordinate_transformer.camera_matrix, 
                                                  self.coordinate_transformer.dist_coefs, 
                                                  None, 
                                                  self.coordinate_transformer.camera_matrix)
                center = self.detect_object_center(undistorted_frame)
                if center is None:
                    centers_world.append(None)
                else:
                    center_array = np.array([[center]], dtype=np.float32)
                    world_pt = cv2.perspectiveTransform(center_array, self.coordinate_transformer.H_inv)
                    world_pt = world_pt[0, 0] * square_size / 10.0
                    centers_world.append(world_pt)
            frame_numbers.append(frame_no)
        cap.release()

        instantaneous_speeds = []
        speed_frame_numbers = []
        for i in range(1, len(centers_world)):
            if centers_world[i] is None or centers_world[i-1] is None:
                instantaneous_speeds.append(None)
            else:
                dx = centers_world[i][0] - centers_world[i-1][0]
                dy = centers_world[i][1] - centers_world[i-1][1]
                dist = math.sqrt(dx * dx + dy * dy)
                speed_m_s = (dist / dt) / 100.0
                instantaneous_speeds.append(speed_m_s)
                speed_frame_numbers.append(frame_numbers[i])
                
        instantaneous_acc = []
        acc_frame_numbers = []
        for i in range(1, len(instantaneous_speeds)):
            if instantaneous_speeds[i] is None or instantaneous_speeds[i-1] is None:
                instantaneous_acc.append(None)
            else:
                a = (instantaneous_speeds[i] - instantaneous_speeds[i-1]) / dt
                instantaneous_acc.append(a)
                acc_frame_numbers.append(speed_frame_numbers[i])
                
        valid_speeds = [s for s in instantaneous_speeds if s is not None]
        valid_acc = [a for a in instantaneous_acc if a is not None]
        avg_speed = np.mean(valid_speeds) if valid_speeds else 0
        avg_acc = np.mean(valid_acc) if valid_acc else 0

        result_str = (
            f"Analyse instantanée de {start_frame} à {end_frame} :\n"
            f"  Vitesse moyenne : {avg_speed:.2f} m/s\n"
            f"  Accélération moyenne : {avg_acc:.2f} m/s²"
        )
        messagebox.showinfo("Analyse de mouvement instantané", result_str)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(speed_frame_numbers, instantaneous_speeds, marker='o', label="Vitesse (m/s)")
        ax1.set_ylabel("Vitesse (m/s)")
        ax1.set_title("Vitesse instantanée vs Cadre")
        ax1.legend()

        ax2.plot(acc_frame_numbers, instantaneous_acc, marker='x', color='red', label="Accélération (m/s²)")
        ax2.set_xlabel("Numéro de cadre")
        ax2.set_ylabel("Accélération (m/s²)")
        ax2.set_title("Accélération instantanée vs Cadre")
        ax2.legend()

        plt.tight_layout()
        plt.show()
