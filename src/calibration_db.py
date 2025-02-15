import json
import os
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CALIB_DB = os.path.join(DATA_DIR, "calibration_db.json")

print(BASE_DIR)
def load_calibrations():
    if os.path.exists(CALIB_DB):
        with open(CALIB_DB, "r") as f:
            return json.load(f)
    else:
        return {}

def save_calibration(camera_name, camera_matrix, dist_coefs, square_size, pattern_size):
    # Create the data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    calibrations = load_calibrations()
    calibrations[camera_name] = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coefs": dist_coefs.tolist(),
        "square_size": square_size,
        "pattern_size": list(pattern_size)
    }
    with open(CALIB_DB, "w") as f:
        json.dump(calibrations, f, indent=4)