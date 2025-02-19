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

# Compute the project root (three levels up)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHOTOS_FOLDER = os.path.join(BASE_DIR, "data\\images")

class Experience(tb.Frame):
    def __init__(self, master):
        super().__init__(master, padding=10)
        self.grid(sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)  # La zone centrale s'étend
        self.bind("<Visibility>", self.on_visibility)
        top_frame = tb.Frame(self)
        top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=1)
        
        # Colonne de gauche : Options caméra et boutons d'actions
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
        
        
        action_buttons_frame = tb.Frame(cam_options_frame)
        action_buttons_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        action_buttons_frame.columnconfigure(0, weight=1)
        action_buttons_frame.columnconfigure(1, weight=1)
        action_buttons_frame.columnconfigure(2, weight=1)
        tb.Button(action_buttons_frame, text="Capture Images from Phone", command=self.capture_video_from_phone, bootstyle="primary")\
            .grid(row=0, column=0, sticky="ew", padx=3, pady=3)
        tb.Button(action_buttons_frame, text="Capture Video from Phone", command=self.capture_video_from_phone, bootstyle="success")\
            .grid(row=0, column=1, sticky="ew", padx=3, pady=3)
        tb.Button(action_buttons_frame, text="Refresh Captured Images", command=self.load_captured_images, bootstyle="warning")\
            .grid(row=0, column=2, sticky="ew", padx=3, pady=3)
            
        self.qr_frame = tb.Labelframe(top_frame, text="QR Code", bootstyle="info")
        self.qr_frame.grid(row=0, column=1, rowspan=1, sticky="nsew", padx=5, pady=5)
        tb.Label(self.qr_frame, text="Le QR Code s'affichera ici", font=("Segoe UI", 12)).pack(padx=10, pady=10)
        
        # --- Zone centrale avec deux blocs côte à côte ---
        center_frame = tb.Frame(self)
        center_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        # La colonne 0 (captured media) aura un poids de 1 et la colonne 1 (additional block) un poids de 3
        center_frame.columnconfigure(0, weight=1)
        center_frame.columnconfigure(1, weight=5)
        center_frame.rowconfigure(0, weight=1)

        # Bloc des images capturées (plus petit en largeur)
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

        # Bouton pour extraire la matrice d'homographie
        extraire_button = tb.Button(captured_frame, text="Extraire la matrice d'homographie", command=self.extract_homography_matrix, bootstyle="info")
        extraire_button.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        # --- Bloc Extraction & Traitement ---
        self.additional_block = tb.Labelframe(center_frame, text="Extraction & Traitement", bootstyle="info")
        self.additional_block.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        # On définit 2 colonnes pour organiser les éléments côte à côte
        self.additional_block.columnconfigure(0, weight=1)
        self.additional_block.columnconfigure(1, weight=1)
        self.additional_block.rowconfigure(2, weight=1)

        # ===== Ligne 0 : Contrôles des sliders (titre et slider côte à côte) =====
        # Contrôles pour la frame de début
        start_controls = tb.Frame(self.additional_block)
        start_controls.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        start_label = tb.Label(start_controls, text="Frame de début :", font=("Segoe UI", 10))
        start_label.pack(side="left", padx=(0, 5))
        self.start_frame_slider = tb.Scale(
            start_controls,
            from_=0,
            to=100,
            orient="horizontal",
            bootstyle="info",
            length=150,   # slider un peu petit
            command=self.update_start_frame_preview
        )
        self.start_frame_slider.pack(side="left")

        # Contrôles pour la frame de fin
        end_controls = tb.Frame(self.additional_block)
        end_controls.grid(row=0, column=1, sticky="w", padx=10, pady=5)
        end_label = tb.Label(end_controls, text="Frame de fin :", font=("Segoe UI", 10))
        end_label.pack(side="left", padx=(0, 5))
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

        # ===== Ligne 1 : Zones d'aperçu des frames côte à côte =====
        self.start_preview_label = tb.Label(self.additional_block, relief="sunken")
        self.start_preview_label.grid(row=1, column=0, padx=10, pady=5)
        self.end_preview_label = tb.Label(self.additional_block, relief="sunken")
        self.end_preview_label.grid(row=1, column=1, padx=10, pady=5)

        # ===== Ligne 2 : Bouton Extraire placé en bas à droite =====
        extract_button = tb.Button(self.additional_block, text="Extraire", command=self.extract_frames, bootstyle="success")
        extract_button.grid(row=3, column=1, sticky="se", padx=10, pady=10)
        extract_button = tb.Button(self.additional_block, text="substruct", command=self.subtract_frames, bootstyle="success")
        extract_button.grid(row=3, column=2, sticky="se", padx=10, pady=10)
        self.calib_files = [] 
        self.thumbnail_images = {} 
        self.current_camera_folder = None
        self.cap = None
        # For video selection/upload
        self.selected_video_path = None  # currently selected video path
        self.uploaded_video_path = None  # stored video path for next treatment
        self.selected_video_button = None
        
        self.frame1_selected = None  # Start frame
        self.frame2_selected = None  # Middle frame
        self.frame3_selected = None  # End frame
        
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
        # Update status if needed

    def toggle_video_selection(self, full_path, button):
        # Only one video can be selected at a time.
        if self.selected_video_path == full_path:
            self.selected_video_path = None
            button.configure(bootstyle="link")
        else:
            self.selected_video_path = full_path
            button.configure(bootstyle="success")
            # Reset previous selection (if any)
            if self.selected_video_button and self.selected_video_button is not button:
                self.selected_video_button.configure(bootstyle="link")
            self.selected_video_button = button
        # Update status if needed

    def capture_video_from_phone(self):
        selected_camera = self.camera_choice.get()
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
        # Largeur disponible dans le canvas
        canvas_width = event.width
        thumb_size = 110  # taille approximative de la miniature + padding
        num_columns = max(1, canvas_width // thumb_size)
        
        # Réorganiser les miniatures
        for idx, widget in enumerate(self.images_frame.winfo_children()):
            row = idx // num_columns
            col = idx % num_columns
            widget.grid_configure(row=row, column=col)
     
    def load_captured_images(self):
        # Clear any existing thumbnails
        for widget in self.images_frame.winfo_children():
            widget.destroy()
        
        camera_name = self.camera_choice.get().strip()
        print("Loading images for camera:", camera_name)
        if camera_name:
            self.ensure_camera_folder(camera_name)
        
        if os.path.exists(self.current_camera_folder):
            files = sorted(os.listdir(self.current_camera_folder))
            image_extensions = ('.jpg', '.jpeg', '.png')
            video_extensions = ('.mp4', '.avi', '.mov')  # add other video formats if needed
            
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
                    # Create a generic video icon if not already done
                    if not hasattr(self, 'video_icon'):
                        video_icon = Image.new('RGB', (100, 100), color='gray')
                        draw = ImageDraw.Draw(video_icon)
                        # Draw a white triangle (play icon)
                        draw.polygon([(35, 25), (35, 75), (75, 50)], fill='white')
                        self.video_icon = ImageTk.PhotoImage(video_icon)
                    self.thumbnail_images[full_path] = self.video_icon
                    btn = tb.Button(self.images_frame, image=self.video_icon, bootstyle="link")
                    btn.config(command=lambda fp=full_path, b=btn: self.toggle_video_selection(fp, b))
                    btn.grid(row=0, column=idx, padx=5, pady=5)
        else:
            messagebox.showerror("Error", f"Photos folder not found at {PHOTOS_FOLDER}.")
    
    def upload_video(self):
        if self.selected_video_path:
            self.uploaded_video_path = self.selected_video_path
            self.cap = cv2.VideoCapture(self.uploaded_video_path)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Mettre à jour la plage des sliders
            self.start_frame_slider.config(to=total_frames - 1)
            self.end_frame_slider.config(to=total_frames - 1)
            print("Selected video for upload:", self.uploaded_video_path)
            messagebox.showinfo("Upload", f"Video selected for upload:\n{self.uploaded_video_path}")
        else:
            messagebox.showwarning("Upload", "Aucune vidéo sélectionnée!")

    def extract_homography_matrix(self):
        """
        Extrait la matrice d'homographie à partir de l'image sélectionnée dans le block Captured Images.
        Les paramètres de calibration (camera matrix, dist_coefs, pattern_size) sont récupérés depuis calibration_db.
        """
        # Vérifier qu'une image a été sélectionnée (ici on utilise la première image sélectionnée)
        if not self.calib_files:
            messagebox.showwarning("Erreur", "Aucune image sélectionnée dans Captured Images.")
            return

        image_path = self.calib_files[0]
        img = cv2.imread(image_path)
        if img is None:
            messagebox.showerror("Erreur", f"Impossible de charger l'image : {image_path}")
            return

        # Récupérer le nom de la caméra sélectionnée dans l'interface
        selected_camera = self.camera_choice.get().strip()
        if not selected_camera:
            messagebox.showerror("Erreur", "Aucune caméra sélectionnée.")
            return

        # Charger les calibrations depuis calibration_db
        calibrations = load_calibrations()
        if selected_camera not in calibrations:
            messagebox.showerror("Erreur", f"Calibration introuvable pour la caméra '{selected_camera}'.")
            return

        # Récupérer les paramètres de calibration pour la caméra sélectionnée
        calibration = calibrations[selected_camera]
        camera_matrix = np.array(calibration["camera_matrix"])  # Assurez-vous que c'est un np.ndarray
        dist_coefs = np.array(calibration["dist_coefs"])
        pattern_size = calibration["pattern_size"]  # Par exemple (4, 4) ou (9, 6)

        # Instanciation du CoordinateTransformer avec les paramètres récupérés
        transformer = CoordinateTransformer(camera_matrix, dist_coefs, pattern_size)

        # Calcul de la matrice d'homographie à partir de l'image (après undistortion)
        if transformer.compute_homography(img):
            # Stocker l'objet pour un usage ultérieur
            self.coordinate_transformer = transformer
            print(self.coordinate_transformer.H_inv)
            messagebox.showinfo("Homographie",
                                f"Homographie calculée avec succès.\n\nMatrice H :\n{transformer.H_inv}")
        else:
            messagebox.showerror("Erreur",
                                "Échec du calcul de la matrice d'homographie.\n"
                                "Vérifiez que l'image contient bien le motif (échiquier) attendu.")

    def calculate_fps(self):
        cap = cv2.VideoCapture(self.uploaded_video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)

        cap.release()
        return fps
    # def extract_frames(self):
    #     if not self.uploaded_video_path:
    #         messagebox.showwarning("Extraction", "Aucune vidéo n'a été sélectionnée!")
    #         return

    #     # Récupérer les valeurs des sliders
    #     start_frame = int(self.start_frame_slider.get())
    #     end_frame = int(self.end_frame_slider.get())

    #     if self.cap is None:
    #         self.cap = cv2.VideoCapture(self.uploaded_video_path)
        
    #     total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     # Vérification des valeurs
    #     if start_frame < 0 or end_frame >= total_frames or start_frame > end_frame:
    #         messagebox.showerror("Erreur", "Valeurs de frames invalides!")
    #         return

    #     # Créer un dossier pour sauvegarder les frames extraites
    #     output_folder = os.path.join(os.path.dirname(self.uploaded_video_path), "extracted_frames")
    #     if not os.path.exists(output_folder):
    #         os.makedirs(output_folder)

    #     # Positionner la capture sur la frame de début
    #     self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    #     current_frame = start_frame

    #     while current_frame <= end_frame:
    #         ret, frame = self.cap.read()
    #         if not ret:
    #             break
    #         frame_filename = os.path.join(output_folder, f"frame_{current_frame}.jpg")
    #         cv2.imwrite(frame_filename, frame)
    #         current_frame += 1

    #     messagebox.showinfo("Extraction Terminée",
    #                         f"Extraction terminée.\nFrames extraites: {current_frame - start_frame}\nDossier: {output_folder}")
    
    def extract_frames(self):
        if not self.uploaded_video_path:
            messagebox.showwarning("Extraction", "Aucune vidéo n'a été sélectionnée!")
            return

        # Retrieve the slider values for the start and end frames
        start_frame = int(self.start_frame_slider.get())
        end_frame = int(self.end_frame_slider.get())

        if self.cap is None:
            self.cap = cv2.VideoCapture(self.uploaded_video_path)

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame < 0 or end_frame >= total_frames or start_frame > end_frame:
            messagebox.showerror("Erreur", "Valeurs de frames invalides!")
            return

        # Create a folder to save the extracted frames
        output_folder = os.path.join(os.path.dirname(self.uploaded_video_path), "extracted_frames")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # --- Extract the Start Frame ---
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = self.cap.read()
        if ret:
            start_frame_filename = os.path.join(output_folder, f"frame_{start_frame}.jpg")
            cv2.imwrite(start_frame_filename, frame)
            self.frame1_selected = frame.copy()
        else:
            messagebox.showerror("Erreur", f"Impossible de lire la frame {start_frame}!")
            return

        # --- Extract the Middle Frame ---
        mid_frame = start_frame + ((end_frame - start_frame) // 2)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = self.cap.read()
        if ret:
            mid_frame_filename = os.path.join(output_folder, f"frame_{mid_frame}.jpg")
            cv2.imwrite(mid_frame_filename, frame)
            self.frame2_selected = frame.copy()
        else:
            messagebox.showerror("Erreur", f"Impossible de lire la frame {mid_frame}!")
            return

        # --- Extract the End Frame ---
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
        ret, frame = self.cap.read()
        if ret:
            end_frame_filename = os.path.join(output_folder, f"frame_{end_frame}.jpg")
            cv2.imwrite(end_frame_filename, frame)
            self.frame3_selected = frame.copy()
        else:
            messagebox.showerror("Erreur", f"Impossible de lire la frame {end_frame}!")
            return

        messagebox.showinfo("Extraction Terminée",
                            f"Extraction terminée.\nFrames extraites: 3\nDossier: {output_folder}")

    def update_start_frame_preview(self, value):
        frame_number = int(float(value))
        if self.cap is None:
            return
        # Positionner la capture sur la frame souhaitée
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        # Obtenir la résolution de la vidéo
        video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Réduire la taille d'affichage (facteur de 0.5 pour moitié de la taille originale)
        scale_factor = 0.2
        display_width = int(video_width * scale_factor)
        display_height = int(video_height * scale_factor)
        
        image = image.resize((display_width, display_height))
        image_tk = ImageTk.PhotoImage(image)
        self.start_preview_label.config(image=image_tk)
        self.start_preview_label.image = image_tk  # Conserver la référence

    def update_end_frame_preview(self, value):
        frame_number = int(float(value))
        if self.cap is None:
            return
        # Positionner la capture sur la frame souhaitée
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        # Obtenir la résolution de la vidéo
        video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Réduire la taille d'affichage (facteur de 0.5 pour moitié de la taille originale)
        scale_factor = 0.2
        display_width = int(video_width * scale_factor)
        display_height = int(video_height * scale_factor)
        
        image = image.resize((display_width, display_height))
        image_tk = ImageTk.PhotoImage(image)
        self.end_preview_label.config(image=image_tk)
        self.end_preview_label.image = image_tk
        
    def subtract_frames(self):
        if (self.frame1_selected is None or 
            self.frame2_selected is None or 
            self.frame3_selected is None):
            self.status_label.config(text="⚠️ ")
            messagebox.showinfo("Upload", "Veuillez extraire les frames d'abord.")

            return

        self.analyze_frame_difference(self.frame1_selected,
                                        self.frame2_selected,
                                        self.frame3_selected,
                                        threshold=30)
    
    def extract_selected_frames(self):

        start_frame = int(self.start_frame_slider.get())
        end_frame = int(self.end_frame_slider.get())

        return start_frame, end_frame
 
    def analyze_frame_difference(self, frame1, frame2, frame3, threshold=30):
        def compute_difference_and_centroids(f1, f2):
            if f1.shape != f2.shape:
                f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]))
            difference = cv2.absdiff(f1, f2)
            gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
            _, thresh_diff = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            highlighted = f1.copy()
            object_centers = []
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(highlighted, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        # Convert pixel coordinates to real-world coordinates if a transformation is provided.
                        if self.coordinate_transformer.H_inv is not None:
                            pt = np.array([[[cx, cy]]], dtype=np.float32)
                            world_pt = cv2.perspectiveTransform(pt, self.coordinate_transformer.H_inv)
                            cx, cy = world_pt[0][0]
                        object_centers.append((cx, cy))
                        cv2.circle(highlighted, (int(cx), int(cy)), 5, (255, 0, 0), -1)
            return highlighted, object_centers

        # Compute differences and centroids between frame pairs.
        highlighted1, centers1 = compute_difference_and_centroids(frame1, frame2)
        highlighted2, centers2 = compute_difference_and_centroids(frame2, frame3)
        highlighted3, centers3 = compute_difference_and_centroids(frame1, frame3)

        # Store the computed centroids in an instance variable.
        self.centers = {
            'Frame 1 et Frame 2': centers1,
            'Frame 2 et Frame 3': centers2,
            'Frame 1 et Frame 3': centers3
        }

        # Define a helper to compute Euclidean distance.
        def euclidean_distance(pt1, pt2):
            return math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

        # Calculate pairwise distances from a list of centers.
        def calculate_pairwise_distances(centers_list):
            dists = []
            if len(centers_list) >= 2:
                for i in range(len(centers_list) - 1):
        
                    dists.append(euclidean_distance(centers_list[i], centers_list[i + 1]) * (float(self.square_size_entry.get().strip())))
            return dists


        self.distances = {
            'Frame 1 et Frame 2': calculate_pairwise_distances(centers1),
            'Frame 2 et Frame 3': calculate_pairwise_distances(centers2),
            'Frame 1 et Frame 3': calculate_pairwise_distances(centers3)
        }

        # Retrieve time intervals (assuming extract_selected_frames returns two intervals).
        interval1, interval2 = self.extract_selected_frames()
        print(interval1, interval2)
        messagebox.showinfo("Intervals", f"Interval1: {interval1}, Interval2: {interval2}")

        # Calculate speeds from distances.
        self.fps_video = self.calculate_fps()
        
        def calculate_speeds(distances, interval1, interval2):
            speeds = []
            for i, dist in enumerate(distances):
                interval1 = interval1 / self.fps_video
                interval2 = interval2 / self.fps_video
                print(interval1, interval2)
                
                if i == 0:
                    speed = dist / interval1 if interval1 != 0 else 0
                elif i == 1:
                    speed = dist / interval2  if interval2 != 0 else 0
                else:
                    speed = dist / (interval1 + interval2) if (interval1 + interval2) != 0 else 0
                speeds.append(speed)
            return speeds

        self.speeds = {}
        for section, dists in self.distances.items():
            self.speeds[section] = calculate_speeds(dists, interval1, interval2)

        # Calculate acceleration using the first speed values from 'Frame 1 et Frame 2' and 'Frame 2 et Frame 3'.
        speeds_section1 = self.speeds.get('Frame 1 et Frame 2', [])
        speeds_section2 = self.speeds.get('Frame 2 et Frame 3', [])
        if speeds_section1 and speeds_section2:
            acceleration = (speeds_section2[0] - speeds_section1[0]) / interval2 if interval2 != 0 else 0
        else:
            messagebox.showerror("Erreur", "Les sections 'Frame 1 et Frame 2' et 'Frame 2 et Frame 3' doivent avoir au moins une vitesse.")
            return

        self.acceleration = acceleration

        results = ""
        for section in self.distances:
            results += f"\n{section}:\n"
            for i, (dist, speed) in enumerate(zip(self.distances[section], self.speeds[section])):
                results += f"Distance {i + 1}: {dist:.2f} pixels, Vitesse: {speed:.2f} mm/s\n"
        results += f"\nAccélération: {acceleration:.2f} mm/s²\n"
        self.results_text = results
        print(results)    