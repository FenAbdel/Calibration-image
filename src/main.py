# src/main.py
import tkinter as tk
from tkinter import ttk
from gui.calibration_frame import CalibrationFrame
from gui.coordinates_frame import CoordinatesFrame
from gui.map_frame import MapFrame

class CalibrationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Camera Calibration & Real-World Coordinates")
        self.geometry("600x500")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.calibration_frame = CalibrationFrame(notebook)
        self.coordinates_frame = CoordinatesFrame(notebook)
        self.map_frame = MapFrame(notebook)

        notebook.add(self.calibration_frame, text="Calibration")
        notebook.add(self.coordinates_frame, text="Coordinates")
        notebook.add(self.map_frame, text="Coordinate Map")

if __name__ == "__main__":
    app = CalibrationApp()
    app.mainloop()
