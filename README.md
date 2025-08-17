# Car - Butterfly Object Detector Web App

A lightweight web application for detecting objects (car - butterfly) in images using a YOLO-based model. Users can upload an image and receive a processed image with bounding boxes and class labels.

[App Preview](https://object-detector-9fn3.onrender.com/)

---

## Features

- Detect multiple object classes in an uploaded image.
- Annotated image output showing detected objects.
- Simple web interface using Flask.
- Configurable detection thresholds (confidence & IoU).
- Containerized with Docker for easy deployment.

---

## Requirements

- Python 3.10+
- Docker
- Git
- Free tier deployment possible on platforms like Render (CPU-based, may have long cold-start times).

---

## Project Structure

```bash
.
├── app.py # Flask app entrypoint
├── model/
| ├── init.py
│ └── best.pt
│ └── model.py # ObjectDetector class
├── requirements.txt # Python dependencies
├── Dockerfile
├── templates/
│ └── index.html # Main web interface
├── static/
│ ├── outputs/ # Annotated images saved here
│ └── images/
└── README.md
```

---

## Setup Instructions (Local via Docker)

1. Clone your repository:

```bash
git clone https://github.com/shahdabbar/object-detector.git
cd YOUR_REPO
```

2. Build the Docker image:

```bash
docker build -t object-detector:latest .
```

3. Run the container:

```bash
docker run --rm -p 8000:8000 -e WEIGHTS_PATH=model/best.pt object-detector:latest
```

4. pen your browser and navigate to:
   http://localhost:8000

---

## Using the Interface

1. Click Choose File and upload an image.
2. Click Detect.
3. The page will display the annotated image and a JSON response of detected objects.

### API Endpoint:

- GET /
  render html template

- POST /predict
  Payload: multipart/form-data with key image
  Response: JSON with detections array and image_url.

- GET /health
  Response: "OK" for health checks.

## Known Issues / Limitations

- Cold start delay on free-tier cloud deployments (up to 50 seconds) due render free instance.

- Memory-heavy: Full-size YOLO models can cause worker timeouts on low-RAM environments.

- Single-threaded inference: Each request is blocking; concurrent uploads may queue.

- Smaller images or lighter models (yolov8n.pt) recommended for faster inference. (I did not have the time to update the model weights, but of course it still needs updates and improvements. For now, though, it's working as needed.)

`This project is for educational and demonstration purposes.`

![App Preview](static/images/Screenshot%202025-08-17%20at%207.44.52 PM.png)
