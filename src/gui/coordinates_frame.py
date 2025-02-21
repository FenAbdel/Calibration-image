# combined_frame.py
import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from calibration_db import load_calibrations
from real_coordinates import CoordinateTransformer

class CombinedFrame(tb.Frame):
    def __init__(self, master):
        super().__init__(master, padding=5)  # Padding réduit pour moins d'espace blanc
        self.master = master
        self.grid(sticky="nsew")
        # Disposition en 2 colonnes pour la calibration et l'image de test
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.bind("<Visibility>", self.on_visibility)
        
        # Chemin de l'image de test sélectionnée
        self.test_image_path = None
        
        # -----------------------
        # Ligne 0 - Calibration de la caméra et Image de test (côte à côte)
        # -----------------------
        # Calibration de la caméra (colonne 0)
        calib_frame = tb.Labelframe(self, text="Calibration de la caméra", bootstyle="info")
        calib_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        calib_frame.columnconfigure(1, weight=1)
        tb.Label(calib_frame, text="Nom de la caméra :", font=("Segoe UI", 12))\
            .grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.camera_choice = tb.Combobox(calib_frame, bootstyle="info")
        self.camera_choice.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.update_camera_choices()
        
        # Image de test (colonne 1)
        test_frame = tb.Labelframe(self, text="Image de test", bootstyle="primary")
        test_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        test_frame.columnconfigure(0, weight=1)
        tb.Button(test_frame, text="Sélectionner une image de test", 
                  command=self.select_test_image, bootstyle="primary-outline")\
            .grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # -----------------------
        # Ligne 1 - Sélection de pixel (occuper toute la largeur)
        # -----------------------
        pixel_frame = tb.Labelframe(self, text="Sélection de pixel", bootstyle="secondary")
        pixel_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        pixel_frame.columnconfigure(0, weight=1)
        
        # Boutons radio pour choisir le mode de sélection
        self.pixel_mode_var = tb.StringVar(value="manual")
        radio_frame = tb.Frame(pixel_frame)
        radio_frame.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        tb.Radiobutton(radio_frame, text="Saisie manuelle", variable=self.pixel_mode_var, value="manual",
                       command=self.toggle_manual_entries).grid(row=0, column=0, padx=5)
        tb.Radiobutton(radio_frame, text="Cliquer sur l'image", variable=self.pixel_mode_var, value="click",
                       command=self.toggle_manual_entries).grid(row=0, column=1, padx=5)
        
        # Saisie manuelle des coordonnées de pixel
        self.manual_frame = tb.Frame(pixel_frame)
        self.manual_frame.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        tb.Label(self.manual_frame, text="Pixel X :", font=("Segoe UI", 10)).grid(row=0, column=0, padx=5, pady=5)
        self.pixel_x_entry = tb.Entry(self.manual_frame, width=5)
        self.pixel_x_entry.grid(row=0, column=1, padx=5, pady=5)
        tb.Label(self.manual_frame, text="Pixel Y :", font=("Segoe UI", 10)).grid(row=0, column=2, padx=5, pady=5)
        self.pixel_y_entry = tb.Entry(self.manual_frame, width=5)
        self.pixel_y_entry.grid(row=0, column=3, padx=5, pady=5)
        
        # -----------------------
        # Ligne 2 - Actions
        # -----------------------
        action_frame = tb.Frame(self)
        action_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        # Trois boutons : Calculer, Afficher la matrice, Exporter la matrice
        action_frame.columnconfigure((0, 1, 2), weight=1)
        tb.Button(action_frame, text="Calculer les coordonnées", command=self.calculate_coordinates,
                  bootstyle="success").grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        tb.Button(action_frame, text="Afficher la matrice", command=self.display_matrix,
                  bootstyle="info").grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        tb.Button(action_frame, text="Exporter la matrice", command=self.export_matrix,
                  bootstyle="warning").grid(row=0, column=2, sticky="ew", padx=5, pady=5)
        
        # -----------------------
        # Ligne 3 - Affichage de la matrice de coordonnées
        # -----------------------
        self.matrix_frame = tb.Labelframe(self, text="Matrice de coordonnées", bootstyle="info")
        self.matrix_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        self.matrix_frame.rowconfigure(0, weight=1)
        self.matrix_frame.columnconfigure(0, weight=1)
        self.matrix_text = tk.Text(self.matrix_frame, wrap="none", height=10)
        self.matrix_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = tk.Scrollbar(self.matrix_frame, command=self.matrix_text.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.matrix_text.config(yscrollcommand=scrollbar.set)
        
        # Configuration responsive des lignes
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=2)
    
    def on_visibility(self, event):
        self.update_camera_choices()
            
    def update_camera_choices(self):
        calibrations = load_calibrations()
        cam_names = list(calibrations.keys())
        self.camera_choice['values'] = cam_names
    
    def select_test_image(self):
        self.test_image_path = filedialog.askopenfilename(
            title="Sélectionner une image de test",
            filetypes=[("Fichiers image", "*.jpg;*.jpeg;*.png")]
        )
        if self.test_image_path:
            messagebox.showinfo("Image de test sélectionnée", f"Sélectionné :\n{self.test_image_path}")
    
    def toggle_manual_entries(self):
        if self.pixel_mode_var.get() == "manual":
            self.manual_frame.grid()
        else:
            self.manual_frame.grid_remove()
    
    def calculate_coordinates(self):
        # Récupération des données de calibration
        calibrations = load_calibrations()
        camera_name = self.camera_choice.get().strip()
        if not camera_name or camera_name not in calibrations:
            messagebox.showerror("Erreur", "Veuillez sélectionner une calibration de caméra valide.")
            return
        cam_data = calibrations[camera_name]
        camera_matrix = np.array(cam_data["camera_matrix"])
        dist_coefs = np.array(cam_data["dist_coefs"])
        pattern_size = tuple(cam_data["pattern_size"])
        
        if not self.test_image_path:
            messagebox.showerror("Erreur", "Veuillez sélectionner une image de test.")
            return
        
        test_img = cv2.imread(self.test_image_path)
        if test_img is None:
            messagebox.showerror("Erreur", "Impossible de charger l'image de test.")
            return
        
        transformer = CoordinateTransformer(camera_matrix, dist_coefs, pattern_size)
        if not transformer.compute_homography(test_img):
            messagebox.showerror("Erreur", "Échec du calcul de l'homographie sur l'image de test.")
            return
        
        undistorted = transformer.undistort_image(test_img)
        
        # Récupération des points sélectionnés
        if self.pixel_mode_var.get() == "manual":
            try:
                x = int(self.pixel_x_entry.get())
                y = int(self.pixel_y_entry.get())
            except ValueError:
                messagebox.showerror("Erreur", "Veuillez saisir des valeurs entières valides pour les coordonnées de pixel.")
                return
            selected_points = [(x, y)]
        else:
            selected_points = self.select_points_with_feedback(undistorted, transformer)
            if not selected_points:
                messagebox.showerror("Erreur", "Aucun point n'a été sélectionné.")
                return
        
        world_coords_list = []
        for (x, y) in selected_points:
            try:
                wc = transformer.pixel_to_world(np.array([[x, y]], dtype=np.float32))
                world_coords_list.append((wc[0, 0], wc[0, 1]))
                # Marquer le point sur l'image
                cv2.circle(undistorted, (x, y), 5, (0, 255, 255), -1)
                coord_text = f"({wc[0,0]:.2f}, {wc[0,1]:.2f})"
                cv2.putText(undistorted, coord_text, (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur en calculant la coordonnée monde pour ({x}, {y}) : {e}")
                return

        # Si le mode interactif est sélectionné, ajouter le système de coordonnées
        if self.pixel_mode_var.get() == "click":
            h, w = undistorted.shape[:2]
            # Choix d'un point d'origine visible (par exemple, en bas à gauche avec une marge)
            origin = (50, h - 50)
            arrow_length = 100  # Longueur des flèches

            # Flèche de l'axe X (en vert)
            cv2.arrowedLine(undistorted, origin, (origin[0] + arrow_length, origin[1]),
                            (0, 255, 0), 3, tipLength=0.1)
            cv2.putText(undistorted, "X", (origin[0] + arrow_length + 10, origin[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            # Flèche de l'axe Y (en rouge)
            cv2.arrowedLine(undistorted, origin, (origin[0], origin[1] - arrow_length),
                            (0, 0, 255), 3, tipLength=0.1)
            cv2.putText(undistorted, "Y", (origin[0] - 30, origin[1] - arrow_length - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("Coordonnées calculées", undistorted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        summary = ""
        for i, ((x, y), (wx, wy)) in enumerate(zip(selected_points, world_coords_list), start=1):
            summary += f"Point {i} - Pixel : ({x}, {y})  Monde : ({wx:.2f}, {wy:.2f})\n"
        messagebox.showinfo("Coordonnées", summary)

    
    def select_points_with_feedback(self, image, transformer):
        """
        Ouvre une fenêtre Matplotlib pour afficher l'image non déformée.
        En déplaçant la souris, une annotation affiche en temps réel les coordonnées pixel et monde.
        Un clic gauche sélectionne un point. Appuyez sur 'Terminer' pour finaliser.
        Renvoie une liste de tuples (x, y) des pixels sélectionnés.
        """
        selected_points = []
        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title("Survolez pour voir les coordonnées.\nCliquez pour sélectionner des points.\nAppuyez sur 'Terminer' lorsque c'est fini.")
        
        annot = ax.annotate("", xy=(0, 0), xytext=(10, 10),
                            textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
                            fontsize=10, color="red")
        annot.set_visible(False)
        
        def on_move(event):
            if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
                x, y = int(event.xdata), int(event.ydata)
                try:
                    world = transformer.pixel_to_world(np.array([[x, y]], dtype=np.float32))
                    world_text = f"Pixel : ({x}, {y})\nMonde : ({world[0,0]:.2f}, {world[0,1]:.2f})"
                except Exception:
                    world_text = f"Pixel : ({x}, {y})\nMonde : Erreur"
                annot.xy = (event.xdata, event.ydata)
                annot.set_text(world_text)
                annot.set_visible(True)
                fig.canvas.draw_idle()
        
        def on_click(event):
            if event.inaxes == ax and event.button == 1 and event.xdata is not None and event.ydata is not None:
                x, y = int(event.xdata), int(event.ydata)
                selected_points.append((x, y))
                ax.plot(x, y, marker='o', color='yellow', markersize=8)
                fig.canvas.draw_idle()
        
        cid_move = fig.canvas.mpl_connect("motion_notify_event", on_move)
        cid_click = fig.canvas.mpl_connect("button_press_event", on_click)
        
        finished = [False]
        def finish(event):
            finished[0] = True
            plt.close(fig)
        
        ax_button = plt.axes([0.8, 0.01, 0.15, 0.05])
        btn = Button(ax_button, 'Terminer')
        btn.on_clicked(finish)
        
        plt.show()
        
        fig.canvas.mpl_disconnect(cid_move)
        fig.canvas.mpl_disconnect(cid_click)
        
        return selected_points

    def export_matrix(self):
        # Fonction pour exporter la matrice dans un fichier
        calibrations = load_calibrations()
        camera_name = self.camera_choice.get().strip()
        if not camera_name or camera_name not in calibrations:
            messagebox.showerror("Erreur", "Veuillez sélectionner une calibration de caméra valide.")
            return
        cam_data = calibrations[camera_name]
        camera_matrix = np.array(cam_data["camera_matrix"])
        dist_coefs = np.array(cam_data["dist_coefs"])
        pattern_size = tuple(cam_data["pattern_size"])
        
        if not self.test_image_path:
            messagebox.showerror("Erreur", "Veuillez sélectionner une image de test.")
            return
        
        test_img = cv2.imread(self.test_image_path)
        if test_img is None:
            messagebox.showerror("Erreur", "Impossible de charger l'image de test.")
            return
        
        transformer = CoordinateTransformer(camera_matrix, dist_coefs, pattern_size)
        if not transformer.compute_homography(test_img):
            messagebox.showerror("Erreur", "Échec du calcul de l'homographie sur l'image de test.")
            return
        
        try:
            world_coords, _, _ = transformer.create_world_coordinates_map(test_img.shape)
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la génération de la matrice de coordonnées : {e}")
            return
        
        matrix_str = np.array2string(world_coords, threshold=np.inf, separator=', ')
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Fichiers texte", "*.txt")],
            title="Enregistrer la matrice de coordonnées"
        )
        if not filepath:
            return
        
        try:
            with open(filepath, 'w') as f:
                f.write(matrix_str)
            messagebox.showinfo("Exportation réussie", f"Matrice exportée vers :\n{filepath}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Échec de l'exportation de la matrice : {e}")
    
    def display_matrix(self):
        # Méthode pour afficher la matrice dans l'interface
        calibrations = load_calibrations()
        camera_name = self.camera_choice.get().strip()
        if not camera_name or camera_name not in calibrations:
            messagebox.showerror("Erreur", "Veuillez sélectionner une calibration de caméra valide.")
            return
        cam_data = calibrations[camera_name]
        camera_matrix = np.array(cam_data["camera_matrix"])
        dist_coefs = np.array(cam_data["dist_coefs"])
        pattern_size = tuple(cam_data["pattern_size"])
        
        if not self.test_image_path:
            messagebox.showerror("Erreur", "Veuillez sélectionner une image de test.")
            return
        
        test_img = cv2.imread(self.test_image_path)
        if test_img is None:
            messagebox.showerror("Erreur", "Impossible de charger l'image de test.")
            return
        
        transformer = CoordinateTransformer(camera_matrix, dist_coefs, pattern_size)
        if not transformer.compute_homography(test_img):
            messagebox.showerror("Erreur", "Échec du calcul de l'homographie sur l'image de test.")
            return
        
        try:
            world_coords, _, _ = transformer.create_world_coordinates_map(test_img.shape)
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la génération de la matrice de coordonnées : {e}")
            return
        
        matrix_str = np.array2string(world_coords, threshold=np.inf, separator=', ')
        self.matrix_text.delete("1.0", tk.END)
        self.matrix_text.insert("1.0", matrix_str)
