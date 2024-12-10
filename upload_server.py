from flask import Flask, request, render_template_string
import os

app = Flask(__name__)

# Directory to save uploaded images
UPLOAD_FOLDER = './images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HTML_TEMPLATE = """
<!doctype html>
<title>Upload Calibration Images</title>
<h1>Upload Calibration Images</h1>
<form method="post" enctype="multipart/form-data">
  <input type="file" name="file" multiple>
  <input type="submit" value="Upload">
</form>
"""

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        uploaded_files = request.files.getlist("file")
        for file in uploaded_files:
            if file:
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)
        return f"Uploaded {len(uploaded_files)} files successfully!"
    return render_template_string(HTML_TEMPLATE)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
