import os
import socket
import base64
import threading
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from OpenSSL import crypto


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES_DIR = os.path.join(BASE_DIR, "data")
CERT_FILE = os.path.join(BASE_DIR, "data\\keys\\cert.pem")
KEY_FILE = os.path.join(BASE_DIR, "data\\keys\\key.pem")
TEMPLATE_DIR = os.path.join(BASE_DIR, "src\\interface_phone")
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data\\images')

if not os.path.exists(FILES_DIR):
    os.makedirs(FILES_DIR)
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

capture_app = Flask(__name__, template_folder=TEMPLATE_DIR)
target_upload_folder = UPLOAD_FOLDER

capture_server_started = False

# ----------------------
# Security Functions
# ----------------------
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
    cert.gmtime_adj_notAfter(365 * 24 * 60 * 60)  # 1 year
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha256')

    # Save the certificate and private key
    with open(CERT_FILE, "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    with open(KEY_FILE, "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))



def get_local_ip(): 
    """Return the local IP address that is likely accessible on the network."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connecting to an external host (doesn't send data)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


# ----------------------
# Flask Routes
# ----------------------

@capture_app.route('/')
def index():
    # Flask will look for index.html in the templates folder (which is set to TEMPLATE_DIR).
    return render_template('index.html')

@capture_app.route('/capture', methods=['POST'])
def capture():
    """Handle image capture requests."""
    try:
        # Use the global target_upload_folder
        global target_upload_folder
        
        image_data = request.json['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Create target directory if it doesn't exist
        if not os.path.exists(target_upload_folder):
            os.makedirs(target_upload_folder)

        filename = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(target_upload_folder, filename)

        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(image_data))

        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ----------------------
# Server Management
# ----------------------
def run_capture_server():
    # Create self-signed certificate if not exists
    if not (os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE)):
        print("Creating self-signed certificate...")
        create_self_signed_cert()
    context = (CERT_FILE, KEY_FILE)
    # Run the Flask server (this call is blocking, so run it in a thread)
    capture_app.run(host='0.0.0.0', port=5000, ssl_context=context, debug=False)

def start_capture_server_in_thread(target_folder):
    global capture_server_started, target_upload_folder
    if target_folder:
        target_upload_folder = target_folder
    if not capture_server_started:
        threading.Thread(target=run_capture_server, daemon=True).start()
        capture_server_started = True

@capture_app.route('/capture_video', methods=['POST'])
def capture_video():
    """Gère la réception d'une vidéo encodée en base64 et l'enregistre dans le format approprié."""
    try:
        global target_upload_folder
        
        video_data = request.json['video']
        mime_type = request.json.get('mimeType', 'video/webm')
        
        if ',' in video_data:
            video_data = video_data.split(',')[1]

        if not os.path.exists(target_upload_folder):
            os.makedirs(target_upload_folder)

        # Déterminer l'extension en fonction du type MIME
        extension = '.mp4' if 'mp4' in mime_type else '.webm'
        filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}{extension}"
        filepath = os.path.join(target_upload_folder, filename)

        # Sauvegarder la vidéo
        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(video_data))

        # Si ce n'est pas déjà un MP4 et que ffmpeg est disponible, convertir en MP4
        if extension != '.mp4' and shutil.which('ffmpeg'):
            output_filepath = filepath.replace(extension, '.mp4')
            try:
                subprocess.run([
                    'ffmpeg', '-i', filepath,
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-crf', '23',
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    output_filepath
                ], check=True, capture_output=True)
                
                # Si la conversion réussit, supprimer le fichier original
                os.remove(filepath)
                filename = os.path.basename(output_filepath)
            except subprocess.CalledProcessError:
                # En cas d'erreur de conversion, garder le fichier original
                pass

        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Ajouter ces imports au début du fichier
import subprocess
import shutil