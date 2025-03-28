from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from torch_utills import transform_image, get_prediction  # Correct import

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the upload folder exists

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

@app.route("/")
def home():
    """Serves the homepage with the drag-and-drop UI."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handles image upload, processes it, and returns model prediction."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)  # Save the uploaded file

        # Transform and predict
        with open(filepath, "rb") as f:
            image_bytes = f.read()
        
        image_tensor = transform_image(image_bytes)  # Convert image to tensor
        prediction = get_prediction(image_tensor)  # Predict digit

        return jsonify({"prediction": prediction.item(), "image_url": filepath})

    return jsonify({"error": "Invalid file format"}), 400

# Route to serve the favicon
@app.route('/favicon.ico')
def favicon():
    favicon_path = os.path.join(app.root_path, "static", "favicon.ico")
    return send_from_directory(os.path.dirname(favicon_path), "favicon.ico", mimetype="image/vnd.microsoft.icon")
    # return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port)
