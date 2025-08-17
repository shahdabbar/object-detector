# model.py
import io
import os
import uuid
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, weights_path="model/best.pt", conf=0.25, iou=0.45, imgsz=640, device=None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weights_path = weights_path
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load model once
        self.model = YOLO(self.weights_path)

    def _image_from_bytes(self, image_bytes: bytes) -> Image.Image:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def predict(self, image_bytes: bytes):
        """
        Returns (detections_list, annotated_image_np)
        detections_list: [{x1,y1,x2,y2,conf,cls,name}, ...]
        annotated_image_np: numpy array (H,W,3) in RGB
        """
        img = self._image_from_bytes(image_bytes)

        # Run inference
        results = self.model.predict(
            img,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )
        r = results[0]

        # Parse detections
        dets = []
        names = r.names
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss  = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                dets.append({
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                    "conf": float(c), "cls": int(k), "name": names.get(int(k), str(int(k)))
                })

        # Render annotated image (Ultralytics gives BGR np array)
        annotated_bgr = r.plot()  # (H,W,3) BGR
        annotated_rgb = annotated_bgr[:, :, ::-1].copy()

        return dets, annotated_rgb

    def save_annotated(self, annotated_rgb: np.ndarray, out_dir: str = "static/outputs") -> str:
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{uuid.uuid4().hex}.jpg"
        fpath = os.path.join(out_dir, fname)

        Image.fromarray(annotated_rgb).save(fpath, format="JPEG", quality=90)
        return fpath
