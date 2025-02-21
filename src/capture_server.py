import os
import socket
import base64
import threading
from datetime import datetime

import cv2
from flask import Flask, render_template, request, jsonify
from OpenSSL import crypto

# ---------------------------
# Configuration and Setup
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES_DIR = os.path.join(BASE_DIR, "data")
# Using os.path.join to support any OS
CERT_FILE = os.path.join(BASE_DIR, "data", "keys", "cert.pem")
KEY_FILE = os.path.join(BASE_DIR, "data", "keys", "key.pem")
TEMPLATE_DIR = os.path.join(BASE_DIR, "src", "interface_phone")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "images")

# Ensure directories exist
if not os.path.exists(FILES_DIR):
    os.makedirs(FILES_DIR)
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

capture_app = Flask(__name__, template_folder=TEMPLATE_DIR)
target_upload_folder = UPLOAD_FOLDER
capture_server_started = False

# ---------------------------
# Security Functions
# ---------------------------
def create_self_signed_cert():
    keys_dir = os.path.dirname(CERT_FILE)
    if not os.path.exists(keys_dir):
        os.makedirs(keys_dir)

    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 2048)

    cert = crypto.X509()
    cert.get_subject().C = "FR"
    cert.get_subject().ST = "State"
    cert.get_subject().L = "City"
    cert.get_subject().O = "Organization"
    cert.get_subject().OU = "Organizational Unit"
    cert.get_subject().CN = socket.gethostname()

    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365 * 24 * 60 * 60)  # 1 year validity
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha256')

    with open(CERT_FILE, "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    with open(KEY_FILE, "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))

def get_local_ip():
    """Return the local IP address that is likely accessible on the network."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

# ---------------------------
# Flask Routes
# ---------------------------
@capture_app.route('/')
def index():
    return render_template('index.html')

@capture_app.route('/capture', methods=['POST'])
def capture():
    """Handle image capture requests."""
    try:
        global target_upload_folder
        image_data = request.json['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        if not os.path.exists(target_upload_folder):
            os.makedirs(target_upload_folder)

        filename = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(target_upload_folder, filename)

        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(image_data))

        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ---------------------------
# Video Conversion Using OpenCV
# ---------------------------
def convert_webm_to_mp4(input_filepath, output_filepath):
    """
    Convert a WebM video to MP4 using OpenCV.
    Reads the input file frame by frame and writes it to a new MP4 file.
    """
    cap = cv2.VideoCapture(input_filepath)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter for MP4 output.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filepath, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print("Conversion completed:", output_filepath)
    return True

@capture_app.route('/capture_video', methods=['POST'])
def capture_video():
    """
    Handle video capture requests.
    The video is received as a base64-encoded string and saved first as WebM.
    Then, it is converted to MP4 using OpenCV.
    """
    try:
        global target_upload_folder
        video_data = request.json['video']
        if ',' in video_data:
            video_data = video_data.split(',')[1]

        if not os.path.exists(target_upload_folder):
            os.makedirs(target_upload_folder)

        # Save the incoming video as a temporary WebM file.
        temp_filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.webm"
        temp_filepath = os.path.join(target_upload_folder, temp_filename)
        with open(temp_filepath, 'wb') as f:
            f.write(base64.b64decode(video_data))

        # Convert the temporary WebM file to MP4.
        mp4_filepath = temp_filepath.rsplit('.', 1)[0] + '.mp4'
        success = convert_webm_to_mp4(temp_filepath, mp4_filepath)
        if success:
            # Optionally, remove the original WebM file.
            os.remove(temp_filepath)
            filename = os.path.basename(mp4_filepath)
        else:
            filename = os.path.basename(temp_filepath)

        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ---------------------------
# Server Management
# ---------------------------
def run_capture_server():
    if not (os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE)):
        print("Creating self-signed certificate...")
        create_self_signed_cert()
    context = (CERT_FILE, KEY_FILE)
    capture_app.run(host='0.0.0.0', port=5000, ssl_context=context, debug=False)

def start_capture_server_in_thread(target_folder):
    global capture_server_started, target_upload_folder
    if target_folder:
        target_upload_folder = target_folder
    if not capture_server_started:
        threading.Thread(target=run_capture_server, daemon=True).start()
        capture_server_started = True

# ---------------------------
# Main Entry Point
# ---------------------------
if __name__ == "__main__":
    run_capture_server()
