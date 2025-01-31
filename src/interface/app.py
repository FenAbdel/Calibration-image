from flask import Flask, render_template, jsonify
import requests
import os
import time

app = Flask(__name__)

# Configure your phone's IP Webcam settings
PHONE_IP = "172.20.10.3"  # Replace with your phone's IP address
CAPTURE_URL = f"http://{PHONE_IP}:8080/photo.jpg"  # URL to capture photo
SAVE_DIR = "../../calibration_image/compressed_images"  # Directory to save the photos

def capture_photo():
    """Capture a photo from the phone's camera."""
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    try:
        response = requests.get(CAPTURE_URL, stream=True)
        if response.status_code == 200:
            # Save the photo
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(SAVE_DIR, f"photo_{timestamp}.jpg")
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            return f"Photo saved: {file_path}"
        else:
            return f"Failed to capture photo. Status code: {response.status_code}"
    except Exception as e:
        return f"Error capturing photo: {e}"

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    """Handle the button click to capture a photo."""
    result = capture_photo()
    return jsonify({'message': result})

if __name__ == '__main__':
    app.run(debug=True)
