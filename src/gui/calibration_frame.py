import os
import shutil
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
from camera_calibration import CameraCalibrator 

# Compute the project root (three levels up)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHOTOS_FOLDER = os.path.join(BASE_DIR, "data\\images")

class CalibrationFrame(tb.Frame):
    def __init__(self, master):
        super().__init__(master, padding=10)
        self.grid(sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)  # La zone centrale s'étend

        # --- Zone supérieure : Options caméra & QR Code ---
        top_frame = tb.Frame(self)
        top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=1)
        
        # Colonne de gauche : Options caméra et boutons d'actions
        cam_options_frame = tb.Labelframe(top_frame, text="Camera Options", bootstyle="info")
        cam_options_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        cam_options_frame.columnconfigure(1, weight=1)
        
        tb.Label(cam_options_frame, text="Camera Name:", font=("Segoe UI", 12)).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.camera_name_entry = tb.Entry(cam_options_frame, font=("Segoe UI", 12))
        self.camera_name_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        tb.Label(cam_options_frame, text="Square Size (mm):", font=("Segoe UI", 12)).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.square_size_entry = tb.Entry(cam_options_frame, font=("Segoe UI", 12))
        self.square_size_entry.insert(0, "30.0")
        self.square_size_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        tb.Label(cam_options_frame, text="Pattern Size (e.g. 4,4):", font=("Segoe UI", 12)).grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.pattern_size_entry = tb.Entry(cam_options_frame, font=("Segoe UI", 12))
        self.pattern_size_entry.insert(0, "4,4")
        self.pattern_size_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

        # Boutons d'actions sous les options de caméra
        action_buttons_frame = tb.Frame(cam_options_frame)
        action_buttons_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        action_buttons_frame.columnconfigure(0, weight=1)
        action_buttons_frame.columnconfigure(1, weight=1)
        action_buttons_frame.columnconfigure(2, weight=1)
        tb.Button(action_buttons_frame, text="Select Calibration Images", command=self.select_calib_images, bootstyle="primary")\
            .grid(row=0, column=0, sticky="ew", padx=3, pady=3)
        tb.Button(action_buttons_frame, text="Capture Images from Phone", command=self.start_capture_server, bootstyle="success")\
            .grid(row=0, column=1, sticky="ew", padx=3, pady=3)
        tb.Button(action_buttons_frame, text="Refresh Captured Images", command=self.load_captured_images, bootstyle="warning")\
            .grid(row=0, column=2, sticky="ew", padx=3, pady=3)
        
        # Colonne de droite : Affichage du QR Code
        self.qr_frame = tb.Labelframe(top_frame, text="QR Code", bootstyle="info")
        self.qr_frame.grid(row=0, column=1, rowspan=1, sticky="nsew", padx=5, pady=5)
        # Affichage initial pour le QR Code
        tb.Label(self.qr_frame, text="Le QR Code s'affichera ici", font=("Segoe UI", 12)).pack(padx=10, pady=10)
        
        # --- Zone centrale : Images capturées ---
        captured_frame = tb.Labelframe(self, text="Captured Images (Cliquez pour sélectionner/désélectionner)", bootstyle="info")
        captured_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.rowconfigure(1, weight=1)
        captured_frame.columnconfigure(0, weight=1)
        captured_frame.rowconfigure(0, weight=1)

        self.images_canvas = tb.Canvas(captured_frame, background="#ffffff", bd=0, highlightthickness=0)
        self.images_canvas.grid(row=0, column=0, sticky="nsew")
        self.vscroll = tb.Scrollbar(captured_frame, orient="vertical", command=self.images_canvas.yview)
        self.vscroll.grid(row=0, column=1, sticky="ns", padx=5, pady=5)
        self.images_canvas.bind("<Configure>", self.on_canvas_configure)
        
        
        self.images_frame = tb.Frame(self.images_canvas)
        self.images_canvas.create_window((0, 0), window=self.images_frame, anchor="nw")
        self.images_frame.bind("<Configure>", lambda e: self.images_canvas.configure(scrollregion=self.images_canvas.bbox("all")))
        
        # --- Zone inférieure : Statut et calibration ---
        bottom_frame = tb.Frame(self)
        bottom_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        bottom_frame.columnconfigure(0, weight=1)
        self.status_label = tb.Label(bottom_frame, text="0 images validées pour calibration.", font=("Segoe UI", 12))
        self.status_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        tb.Button(bottom_frame, text="Calibrate Camera", command=self.calibrate_camera, bootstyle="success-outline")\
            .grid(row=0, column=1, sticky="e", padx=5, pady=5)
        
        # Variables internes
        self.calib_files = []  # Liste des images sélectionnées pour la calibration
        self.thumbnail_images = {}  # Références vers les PhotoImage pour éviter le garbage collector
        self.current_camera_folder = None
        
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

    # Méthode modifiée pour copier les images sélectionnées dans le dossier de la caméra
    def select_calib_images(self):
        files = filedialog.askopenfilenames(
            title="Select Calibration Images",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
        )
        if not files:
            return

        camera_name = self.camera_name_entry.get().strip()
        if not camera_name:
            messagebox.showerror("Error", "Veuillez renseigner un nom de caméra.")
            return

        self.ensure_camera_folder(camera_name)

        for file in files:
            filename = os.path.basename(file)
            dest_path = os.path.join(self.current_camera_folder, filename)
            try:
                shutil.copy(file, dest_path)
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de la copie de {filename}: {e}")
                continue

            if dest_path not in self.calib_files:
                self.calib_files.append(dest_path)

        messagebox.showinfo("Images Selected", f"{len(files)} images ajoutées.")
        self.update_status_label()
        self.load_captured_images()

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

    def load_captured_images(self):
        # On vide les miniatures précédentes
        for widget in self.images_frame.winfo_children():
            widget.destroy()
        
        camera_name = self.camera_name_entry.get().strip()
        if camera_name:
            self.ensure_camera_folder(camera_name)
        
        if os.path.exists(self.current_camera_folder):
            files = sorted(os.listdir(self.current_camera_folder))
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for idx, filename in enumerate(image_files):
                full_path = os.path.join(self.current_camera_folder, filename)
                try:
                    img = Image.open(full_path)
                    img.thumbnail((100, 100))
                    photo = ImageTk.PhotoImage(img)
                    self.thumbnail_images[full_path] = photo
                    style = "success" if full_path in self.calib_files else "link"
                    btn = tb.Button(self.images_frame, image=photo, bootstyle=style)
                    btn.config(command=lambda fp=full_path, b=btn: self.toggle_image_selection(fp, b))
                    # On place initialement (on pourra réorganiser via on_canvas_configure)
                    btn.grid(row=0, column=idx, padx=5, pady=5)
                except Exception as e:
                    print("Erreur lors du chargement de l'image :", full_path, e)

        else:
            messagebox.showerror("Error", f"Photos folder not found at {PHOTOS_FOLDER}.")

    def toggle_image_selection(self, full_path, button):
        if full_path in self.calib_files:
            self.calib_files.remove(full_path)
            button.configure(bootstyle="link")
        else:
            self.calib_files.append(full_path)
            button.configure(bootstyle="success")
        self.update_status_label()

    def update_status_label(self):
        self.status_label.config(text=f"{len(self.calib_files)} images validées pour calibration.")

    def calibrate_camera(self):
        camera_name = self.camera_name_entry.get().strip()
        if not camera_name:
            messagebox.showerror("Error", "Veuillez renseigner un nom de caméra.")
            return
        try:
            square_size = float(self.square_size_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Taille de carré invalide.")
            return
        try:
            pattern_size = tuple(map(int, self.pattern_size_entry.get().split(',')))
        except Exception:
            messagebox.showerror("Error", "Taille du motif invalide. Format attendu : 4,4")
            return
        if not self.calib_files:
            messagebox.showerror("Error", "Aucune image validée pour la calibration.")
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
            messagebox.showerror("Error", "Calibration échouée : aucune image valide.")
            return
        img = cv2.imread(self.calib_files[0])
        h, w = img.shape[:2]
        ret, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, (w, h), None, None
        )
        save_calibration(camera_name, camera_matrix, dist_coefs, square_size, pattern_size)
        messagebox.showinfo("Succès", f"Caméra '{camera_name}' calibrée et enregistrée.")

    def start_capture_server(self):
        camera_name = self.camera_name_entry.get().strip()
        if not camera_name:
            messagebox.showerror("Error", "Veuillez renseigner un nom de caméra.")
            return
        
        camera_folder = self.ensure_camera_folder(camera_name)
        start_capture_server_in_thread(target_folder=camera_folder)
        
        local_ip = get_local_ip()
        url = f"https://{local_ip}:5000"
        qr = qrcode.QRCode(version=1, box_size=5, border=4)
        qr.add_data(url)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_img_tk = ImageTk.PhotoImage(qr_img)
        
        # Mise à jour de la zone QR Code
        for widget in self.qr_frame.winfo_children():
            widget.destroy()
        tb.Label(self.qr_frame, text="Scannez ce QR code avec votre téléphone :", font=("Segoe UI", 8, "bold"))\
            .pack(padx=10, pady=2)
        tb.Label(self.qr_frame, image=qr_img_tk).pack(padx=10, pady=5)
        tb.Label(self.qr_frame, text=url, font=("Segoe UI", 12))\
            .pack(padx=10, pady=2)
        self.qr_frame.qr_img_tk = qr_img_tk  # Référence pour éviter le garbage collection
