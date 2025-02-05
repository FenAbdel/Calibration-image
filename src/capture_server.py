# src/capture_server.py
import os
import socket
import base64
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from OpenSSL import crypto

# Determine the base directory (project root) by moving one level up from the current file.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Determine the templates folder path (which is also at the project root level in the "templates" folder).
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# Create the Flask app, pointing it to the correct templates folder.
capture_app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Set the upload folder to be at the project root, not inside src.
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'photos')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def create_self_signed_cert():
    # Generate a self-signed certificate
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

    with open("cert.pem", "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    with open("key.pem", "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))

@capture_app.route('/')
def index():
    # Flask will look for index.html in the templates folder (which is set to TEMPLATE_DIR).
    return render_template('index.html')

@capture_app.route('/capture', methods=['POST'])
def capture():
    try:
        image_data = request.json['image']
        # Remove the prefix if it exists ("data:image/jpeg;base64,")
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        filename = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(image_data))

        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def run_capture_server():
    # Create self-signed certificate if not exists
    if not (os.path.exists("cert.pem") and os.path.exists("key.pem")):
        print("Creating self-signed certificate...")
        create_self_signed_cert()
    context = ('cert.pem', 'key.pem')
    # Run the Flask server (this call is blocking, so run it in a thread)
    capture_app.run(host='0.0.0.0', port=5000, ssl_context=context, debug=False)

# Global flag to avoid multiple server threads
capture_server_started = False

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

def start_capture_server_in_thread():
    global capture_server_started
    if not capture_server_started:
        threading.Thread(target=run_capture_server, daemon=True).start()
        capture_server_started = True
