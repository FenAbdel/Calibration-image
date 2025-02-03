# app.py
from flask import Flask, render_template, request, jsonify
import os
import base64
from datetime import datetime
from OpenSSL import SSL

app = Flask(__name__)

# Dossier pour sauvegarder les photos
UPLOAD_FOLDER = 'photos'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Création du certificat auto-signé
def create_self_signed_cert():
    from OpenSSL import crypto
    from socket import gethostname
    
    # Génération de la clé
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 2048)
    
    # Création du certificat
    cert = crypto.X509()
    cert.get_subject().C = "FR"
    cert.get_subject().ST = "State"
    cert.get_subject().L = "City"
    cert.get_subject().O = "Organization"
    cert.get_subject().OU = "Organizational Unit"
    cert.get_subject().CN = gethostname()
    
    # Validité du certificat (1 an)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365*24*60*60)
    
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha256')
    
    # Sauvegarde des fichiers
    with open("cert.pem", "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    with open("key.pem", "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    try:
        # Récupérer l'image en base64 depuis la requête
        image_data = request.json['image']
        # Enlever le préfixe "data:image/jpeg;base64,"
        image_data = image_data.split(',')[1]
        
        # Générer un nom de fichier unique avec la date
        filename = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Décoder et sauvegarder l'image
        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(image_data))
            
        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Installer les dépendances si nécessaire
    try:
        import OpenSSL
    except ImportError:
        print("Installation des dépendances nécessaires...")
        os.system('pip install pyOpenSSL')
    
    # Créer le certificat s'il n'existe pas
    if not (os.path.exists("cert.pem") and os.path.exists("key.pem")):
        print("Création du certificat auto-signé...")
        create_self_signed_cert()
    
    # Démarrer le serveur en HTTPS
    context = ('cert.pem', 'key.pem')
    app.run(host='0.0.0.0', port=5000, ssl_context=context, debug=True)