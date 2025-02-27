import tkinter as tk
from tkinter import ttk
from gui.calibration_frame import CalibrationFrame
from gui.coordinates_frame import CombinedFrame
from gui.experience import Experience

class CalibrationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Camera Calibration & Real-World Coordinates")
        self.geometry("1300x750")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.calibration_frame = CalibrationFrame(notebook)
        self.coordinates_frame = CombinedFrame(notebook)

        self.experience = Experience(notebook)
        
        notebook.add(self.calibration_frame, text="Calibration")
        notebook.add(self.coordinates_frame, text="Coordinates")
        notebook.add(self.experience, text="Experience")


if __name__ == "__main__":
    app = CalibrationApp()
    app.mainloop()
