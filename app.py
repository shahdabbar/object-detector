# app.py
import os
from flask import Flask, render_template, request, jsonify
from model.model import ObjectDetector

WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "model/best.pt")
CONF = float(os.environ.get("CONF", "0.25"))
IOU  = float(os.environ.get("IOU", "0.45"))
IMGZ = int(os.environ.get("IMG_SIZE", "640"))

app = Flask(__name__)

# # Instantiate the detector once at startup
# detector = ObjectDetector(weights_path=WEIGHTS_PATH, conf=CONF, iou=IOU, imgsz=IMGZ)

detector = None  # lazy-loaded detector

@app.before_first_request
def load_detector():
    global detector
    if detector is None:
        print("Loading ObjectDetector...")
        detector = ObjectDetector(weights_path=WEIGHTS_PATH, conf=CONF, iou=IOU, imgsz=IMGZ)
        print("ObjectDetector loaded!")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file part 'image'"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    image_bytes = file.read()

    # run inference
    dets, annotated_rgb = detector.predict(image_bytes)

    # save annotated image
    out_path = detector.save_annotated(annotated_rgb, out_dir="static/outputs")
    # URL to serve in browser
    image_url = f"/{out_path}" if out_path.startswith("static/") else out_path

    return jsonify({"detections": dets, "image_url": image_url})

@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    # for local debug only; in Docker we use gunicorn
    app.run(host="0.0.0.0", port=8000, debug=True)
