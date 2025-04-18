from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from ocr_utils import process_image

app = Flask(__name__)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Configure template and static folders
app.template_folder = "templates"
app.static_folder = "static"

@app.route('/', methods=['GET'])
def get_form():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_images():
    uploaded_files = request.files.getlist('files')
    results = []

    for file in uploaded_files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_DIR, filename)
        file.save(file_path)

        result = process_image(file_path)
        results.append(result)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
