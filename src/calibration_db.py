# src/calibration_db.py
import json
import os
import numpy as np

CALIB_DB = "calibration_db.json"

def load_calibrations():
    if os.path.exists(CALIB_DB):
        with open(CALIB_DB, "r") as f:
            return json.load(f)
    else:
        return {}

def save_calibration(camera_name, camera_matrix, dist_coefs, square_size, pattern_size):
    calibrations = load_calibrations()
    calibrations[camera_name] = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coefs": dist_coefs.tolist(),
        "square_size": square_size,
        "pattern_size": list(pattern_size)
    }
    with open(CALIB_DB, "w") as f:
        json.dump(calibrations, f, indent=4)
