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
        super().__init__(master, padding=20)
        self.master = master
        self.grid(sticky="nsew")
        self.columnconfigure(0, weight=1)
        
        # Variable to store the selected test image path
        self.test_image_path = None
        
        # -----------------------
        # Camera Calibration Section
        # -----------------------
        calib_frame = tb.Labelframe(self, text="Camera Calibration", bootstyle="info")
        calib_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        calib_frame.columnconfigure(1, weight=1)
        tb.Label(calib_frame, text="Camera Name:", font=("Segoe UI", 12))\
            .grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.camera_choice = tb.Combobox(calib_frame, bootstyle="info")
        self.camera_choice.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.update_camera_choices()
        
        # -----------------------
        # Test Image Section
        # -----------------------
        test_frame = tb.Labelframe(self, text="Test Image", bootstyle="primary")
        test_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        test_frame.columnconfigure(0, weight=1)
        tb.Button(test_frame, text="Select Test Image", command=self.select_test_image, bootstyle="primary-outline")\
            .grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # -----------------------
        # Pixel Selection Section
        # -----------------------
        pixel_frame = tb.Labelframe(self, text="Pixel Selection", bootstyle="secondary")
        pixel_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        pixel_frame.columnconfigure(0, weight=1)
        
        # Radio buttons for pixel selection mode
        self.pixel_mode_var = tb.StringVar(value="manual")
        radio_frame = tb.Frame(pixel_frame)
        radio_frame.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        tb.Radiobutton(radio_frame, text="Manual Entry", variable=self.pixel_mode_var, value="manual",
                       command=self.toggle_manual_entries).grid(row=0, column=0, padx=5)
        tb.Radiobutton(radio_frame, text="Click on Image", variable=self.pixel_mode_var, value="click",
                       command=self.toggle_manual_entries).grid(row=0, column=1, padx=5)
        
        # Manual entry for pixel coordinates
        self.manual_frame = tb.Frame(pixel_frame)
        self.manual_frame.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        tb.Label(self.manual_frame, text="Pixel X:", font=("Segoe UI", 10)).grid(row=0, column=0, padx=5, pady=5)
        self.pixel_x_entry = tb.Entry(self.manual_frame, width=5)
        self.pixel_x_entry.grid(row=0, column=1, padx=5, pady=5)
        tb.Label(self.manual_frame, text="Pixel Y:", font=("Segoe UI", 10)).grid(row=0, column=2, padx=5, pady=5)
        self.pixel_y_entry = tb.Entry(self.manual_frame, width=5)
        self.pixel_y_entry.grid(row=0, column=3, padx=5, pady=5)
        
        # -----------------------
        # Actions Section
        # -----------------------
        action_frame = tb.Frame(self)
        action_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
        action_frame.columnconfigure((0, 1), weight=1)
        tb.Button(action_frame, text="Calculate Coordinates", command=self.calculate_coordinates,
                  bootstyle="success").grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        tb.Button(action_frame, text="Export Coordinate Matrix", command=self.export_matrix,
                  bootstyle="warning").grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        # Make rows expand equally for a responsive layout
        for i in range(4):
            self.rowconfigure(i, weight=1)
    
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
            messagebox.showinfo("Test Image Selected", f"Selected:\n{self.test_image_path}")
    
    def toggle_manual_entries(self):
        if self.pixel_mode_var.get() == "manual":
            self.manual_frame.grid()
        else:
            self.manual_frame.grid_remove()
    
    def calculate_coordinates(self):
        # Get camera calibration data
        calibrations = load_calibrations()
        camera_name = self.camera_choice.get().strip()
        if not camera_name or camera_name not in calibrations:
            messagebox.showerror("Error", "Please select a valid camera calibration.")
            return
        cam_data = calibrations[camera_name]
        camera_matrix = np.array(cam_data["camera_matrix"])
        dist_coefs = np.array(cam_data["dist_coefs"])
        pattern_size = tuple(cam_data["pattern_size"])
        
        # Ensure a test image has been selected
        if not self.test_image_path:
            messagebox.showerror("Error", "Please select a test image.")
            return
        
        test_img = cv2.imread(self.test_image_path)
        if test_img is None:
            messagebox.showerror("Error", "Could not load the test image.")
            return
        
        transformer = CoordinateTransformer(camera_matrix, dist_coefs, pattern_size)
        if not transformer.compute_homography(test_img):
            messagebox.showerror("Error", "Failed to compute homography on the test image.")
            return
        
        undistorted = transformer.undistort_image(test_img)
        
        if self.pixel_mode_var.get() == "manual":
            try:
                x = int(self.pixel_x_entry.get())
                y = int(self.pixel_y_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Enter valid integer values for pixel coordinates.")
                return
            selected_points = [(x, y)]
        else:
            # Use the new interactive multi-point selection with real-time feedback
            selected_points = self.select_points_with_feedback(undistorted, transformer)
            if not selected_points:
                messagebox.showerror("Error", "No points were selected.")
                return
        
        # Compute world coordinates for each selected point
        world_coords_list = []
        for (x, y) in selected_points:
            try:
                wc = transformer.pixel_to_world(np.array([[x, y]], dtype=np.float32))
                world_coords_list.append((wc[0, 0], wc[0, 1]))
                # Draw the marker on the undistorted image
                cv2.circle(undistorted, (x, y), 5, (0, 255, 255), -1)
                coord_text = f"({wc[0,0]:.2f}, {wc[0,1]:.2f})"
                cv2.putText(undistorted, coord_text, (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            except Exception as e:
                messagebox.showerror("Error", f"Error computing world coordinate for ({x}, {y}): {e}")
                return
        
        cv2.imshow("Calculated Coordinates", undistorted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Build a summary text of the selected coordinates
        summary = ""
        for i, ((x, y), (wx, wy)) in enumerate(zip(selected_points, world_coords_list), start=1):
            summary += f"Point {i} - Pixel: ({x}, {y})  World: ({wx:.2f}, {wy:.2f})\n"
        messagebox.showinfo("Coordinates", summary)
    
    def select_points_with_feedback(self, image, transformer):
        """
        Opens a Matplotlib window that shows the undistorted image.
        As the user moves the mouse, a live annotation displays the pixel and corresponding world coordinates.
        Left-clicking adds a marker (supporting multi-point selection).
        Press the "Finish" button to end selection.
        Returns a list of (x, y) tuples representing the selected pixel coordinates.
        """
        selected_points = []

        fig, ax = plt.subplots()
        # Display the image (convert BGR to RGB for Matplotlib)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title("Hover to see coordinates. Click to select points. Press 'Finish' when done.")
        
        # Create an annotation for real-time coordinate feedback
        annot = ax.annotate("", xy=(0, 0), xytext=(10, 10),
                            textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
                            fontsize=10, color="red")
        annot.set_visible(False)
        
        def on_move(event):
            if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
                x, y = int(event.xdata), int(event.ydata)
                try:
                    world = transformer.pixel_to_world(np.array([[x, y]], dtype=np.float32))
                    world_text = f"Pixel: ({x}, {y})\nWorld: ({world[0,0]:.2f}, {world[0,1]:.2f})"
                except Exception:
                    world_text = f"Pixel: ({x}, {y})\nWorld: Error"
                annot.xy = (event.xdata, event.ydata)
                annot.set_text(world_text)
                annot.set_visible(True)
                fig.canvas.draw_idle()
        
        def on_click(event):
            if event.inaxes == ax and event.button == 1 and event.xdata is not None and event.ydata is not None:
                x, y = int(event.xdata), int(event.ydata)
                selected_points.append((x, y))
                # Mark the selected point with a yellow circle
                ax.plot(x, y, marker='o', color='yellow', markersize=8)
                fig.canvas.draw_idle()
        
        cid_move = fig.canvas.mpl_connect("motion_notify_event", on_move)
        cid_click = fig.canvas.mpl_connect("button_press_event", on_click)
        
        # Add a "Finish" button using Matplotlib's widget
        finished = [False]  # mutable flag
        
        def finish(event):
            finished[0] = True
            plt.close(fig)
        
        # Place the button in a new axes area
        ax_button = plt.axes([0.8, 0.01, 0.15, 0.05])
        btn = Button(ax_button, 'Finish')
        btn.on_clicked(finish)
        
        plt.show()
        
        # Disconnect callbacks once done
        fig.canvas.mpl_disconnect(cid_move)
        fig.canvas.mpl_disconnect(cid_click)
        
        return selected_points

    def export_matrix(self):
        # Get camera calibration data
        calibrations = load_calibrations()
        camera_name = self.camera_choice.get().strip()
        if not camera_name or camera_name not in calibrations:
            messagebox.showerror("Error", "Please select a valid camera calibration.")
            return
        cam_data = calibrations[camera_name]
        camera_matrix = np.array(cam_data["camera_matrix"])
        dist_coefs = np.array(cam_data["dist_coefs"])
        pattern_size = tuple(cam_data["pattern_size"])
        
        # Ensure a test image has been selected
        if not self.test_image_path:
            messagebox.showerror("Error", "Please select a test image.")
            return
        
        test_img = cv2.imread(self.test_image_path)
        if test_img is None:
            messagebox.showerror("Error", "Could not load the test image.")
            return
        
        transformer = CoordinateTransformer(camera_matrix, dist_coefs, pattern_size)
        if not transformer.compute_homography(test_img):
            messagebox.showerror("Error", "Failed to compute homography on the test image.")
            return
        
        try:
            # Create the coordinate matrix (world_coords) using the image dimensions
            world_coords, _, _ = transformer.create_world_coordinates_map(test_img.shape)
        except Exception as e:
            messagebox.showerror("Error", f"Error generating coordinate matrix: {e}")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")],
            title="Save Coordinate Matrix"
        )
        if not filepath:
            return
        
        try:
            # Convert the matrix to a string and write to file
            array_str = np.array2string(world_coords, threshold=np.inf, separator=', ')
            with open(filepath, 'w') as f:
                f.write(array_str)
            messagebox.showinfo("Export Successful", f"Coordinate matrix exported to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export matrix: {e}")

