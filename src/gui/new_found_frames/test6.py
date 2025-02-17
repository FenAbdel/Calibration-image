import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Entry, Button, Scale, Text, Scrollbar
from PIL import Image, ImageTk
import matplotlib.pyplot as plt


# Variables globales
video_path = None
cap = None
frame1_selected = None
frame2_selected = None

def select_video():
    global video_path, cap
    file_path = filedialog.askopenfilename(
        title="Sélectionner une vidéo",
        filetypes=[("Fichiers vidéo", "*.*")]
    )

    if file_path:
        video_path = file_path
        entry_path.delete(0, tk.END)
        entry_path.insert(0, video_path)
        process_video()

def process_video():
    global cap, total_frames

    if not video_path:
        status_label.config(text="⚠️ Veuillez sélectionner une vidéo d'abord.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        status_label.config(text="❌ Erreur : Impossible d'ouvrir la vidéo.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    status_label.config(text=f"📹 Vidéo chargée ({fps} FPS, {total_frames} frames)")

    slider_frame1.config(from_=0, to=total_frames - 1)
    slider_frame2.config(from_=0, to=total_frames - 1)
    slider_frame1.set(0)
    slider_frame2.set(min(50, total_frames - 1))

    update_frame_preview(0, label_frame1)
    update_frame_preview(min(50, total_frames - 1), label_frame2)

def update_frame_preview(frame_idx, label):
    if not cap:
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()

    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((300, 200), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        label.config(image=img_tk)
        label.image = img_tk

def extract_selected_frames():
    global cap, frame1_selected, frame2_selected

    if not cap:
        status_label.config(text="⚠️ Veuillez charger une vidéo d'abord.")
        return

    frame_idx1 = int(slider_frame1.get())
    frame_idx2 = int(slider_frame2.get())

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx1)
    ret1, frame1 = cap.read()

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx2)
    ret2, frame2 = cap.read()

    if ret1 and ret2:
        save_frames(frame1, frame2)
        frame1_selected, frame2_selected = frame1, frame2
        status_label.config(text="✅ Frames extraites et enregistrées.")
    else:
        status_label.config(text="❌ Erreur lors de l'extraction des frames.")

def save_frames(frame1, frame2):
    folder_name = "captured_frames"
    os.makedirs(folder_name, exist_ok=True)

    path1 = os.path.join(folder_name, "frame_1.jpg")
    path2 = os.path.join(folder_name, "frame_2.jpg")

    cv2.imwrite(path1, frame1)
    cv2.imwrite(path2, frame2)

def subtract_frames():
    global frame1_selected, frame2_selected

    if frame1_selected is None or frame2_selected is None:
        status_label.config(text="⚠️ Veuillez extraire les frames d'abord.")
        return

    # Appeler la nouvelle fonction pour analyser la différence entre les frames
    analyze_frame_difference(frame1_selected, frame2_selected, threshold=30)


# def find_center_of_mass(binary_diff):
#     """Trouve le centre de masse des pixels actifs (valeur 1)."""
#     y_coords, x_coords = np.where(binary_diff == 1)

#     if len(x_coords) > 0 and len(y_coords) > 0:
#         center_x = int(np.mean(x_coords))
#         center_y = int(np.mean(y_coords))
#         return center_x, center_y
#     return None, None


# def display_matrix(matrix):
#     """Affiche la matrice de pixels dans l'interface et l'enregistre dans un fichier texte."""
#     # Affichage de la matrice de pixels dans le widget Text
#     text_matrix.delete("1.0", tk.END)
#     for row in matrix:
#         text_matrix.insert(tk.END, " ".join(map(str, row)) + "\n")
    
#     # Enregistrement de la matrice dans un fichier texte
#     try:
#         # Définir le nom du fichier
#         filename = "matrice_pixels.txt"
        
#         # Obtenir le chemin absolu du répertoire du script
#         script_dir = os.path.dirname(os.path.abspath(__file__))
        
#         # Chemin complet du fichier
#         file_path = os.path.join(script_dir, filename)
        
#         # Utiliser numpy.savetxt pour enregistrer la matrice
#         np.savetxt(file_path, matrix, fmt='%d', delimiter=' ')
        
#         print(f"La matrice a été enregistrée avec succès dans le fichier : {file_path}")
#     except Exception as e:
#         print(f"Une erreur s'est produite lors de l'enregistrement de la matrice : {e}")

def analyze_frame_difference(frame1, frame2, threshold=30):
    # Vérifier si les frames sont valides
    if frame1 is None or frame2 is None:
        raise ValueError("L'une ou les deux frames sont invalides.")
    
    # S'assurer que les frames ont les mêmes dimensions
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    
    # Calculer la différence absolue
    difference = cv2.absdiff(frame1, frame2)
    
    # Convertir la différence en niveaux de gris et appliquer un seuillage
    gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, thresh_diff = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Trouver les contours des différences significatives
    contours, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Copier la première frame pour l'affichage
    highlighted = frame1.copy()
    object_centers = []  # Stocker les centres des objets détectés

    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Ignorer les petits contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(highlighted, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Calculer le centroïde de l'objet
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                object_centers.append((cx, cy))
                cv2.circle(highlighted, (cx, cy), 5, (255, 0, 0), -1)  # Dessiner le centroïde

    print("Centres des objets détectés :", object_centers)
    
    # Convertir les images BGR en RGB pour Matplotlib
    frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    highlighted_rgb = cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB)
    thresh_diff_rgb = cv2.cvtColor(thresh_diff, cv2.COLOR_GRAY2RGB)
    
    # Afficher les images
    plt.figure(figsize=(15, 10))
    
    plt.subplot(221), plt.imshow(frame1_rgb)
    plt.title('Frame 1'), plt.axis('off')
    
    plt.subplot(222), plt.imshow(frame2_rgb)
    plt.title('Frame 2'), plt.axis('off')
    
    plt.subplot(223), plt.imshow(thresh_diff_rgb)
    plt.title('Différence seuillée'), plt.axis('off')
    
    plt.subplot(224), plt.imshow(highlighted_rgb)
    plt.title('Différences avec centroïdes'), plt.axis('off')
    # Enregistrement des coordonnées des centroïdes dans un fichier texte
    with open("centroides.txt", "w") as file:
        for center in object_centers:
            file.write(f"{center[0]} {center[1]}\n")
    
    plt.tight_layout()
    plt.show()

        
def afficher_matrice_en_image(fichier_matrice):
    """Affiche la matrice de pixels stockée dans un fichier en tant qu'image."""
    try:
        # Charger la matrice depuis le fichier
        matrice = np.loadtxt(fichier_matrice, dtype=int)
        
        # Afficher la matrice en tant qu'image
        plt.imshow(matrice, cmap='gray', interpolation='nearest')
        plt.title("Matrice de Pixels")
        plt.colorbar()  # Ajouter une barre de couleur pour référence
        plt.show()
    except Exception as e:
        print(f"Une erreur s'est produite lors de l'affichage de la matrice : {e}")

root = tk.Tk()
root.title("Sélection de Frames Vidéo")
root.geometry("900x700")

Label(root, text="Chemin de la vidéo :").pack()
entry_path = Entry(root, width=50)
entry_path.pack()
Button(root, text="📂 Parcourir", command=select_video).pack()

Label(root, text="Sélectionnez la 1ère frame :").pack()
slider_frame1 = Scale(root, from_=0, to=100, orient="horizontal", length=500, command=lambda val: update_frame_preview(int(val), label_frame1))
slider_frame1.pack()

Label(root, text="Sélectionnez la 2ème frame :").pack()
slider_frame2 = Scale(root, from_=0, to=100, orient="horizontal", length=500, command=lambda val: update_frame_preview(int(val), label_frame2))
slider_frame2.pack()

Button(root, text="📸 Extraire et enregistrer", command=extract_selected_frames, bg="lightblue").pack()

frame_display = tk.Frame(root)
frame_display.pack()

label_frame1 = Label(frame_display)
label_frame1.pack(side="left", padx=10)

label_frame2 = Label(frame_display)
label_frame2.pack(side="right", padx=10)

Button(root, text="➖ Soustraction des frames", command=subtract_frames, bg="lightgreen").pack()

Label(root, text="Matrice des pixels après soustraction :").pack()

text_frame = tk.Frame(root)
text_frame.pack(fill="both", expand=True)


scrollbar = Scrollbar(text_frame)
scrollbar.pack(side="right", fill="y")

text_matrix = Text(text_frame, wrap="none", height=15, width=100, yscrollcommand=scrollbar.set)
text_matrix.pack()

scrollbar.config(command=text_matrix.yview)

status_label = Label(root, text="", fg="red")
status_label.pack()

root.mainloop()
