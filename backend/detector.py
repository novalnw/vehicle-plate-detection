from pathlib import Path

import cv2
from ultralytics import YOLO

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class VehicleDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, image):
        results = self.model(image, verbose=False)
        vehicles = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in VEHICLE_CLASSES:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    vehicles.append(
                        {
                            "bbox": [x1, y1, x2, y2],
                            "label": VEHICLE_CLASSES[cls_id],
                            "confidence": round(float(box.conf[0]), 4),
                        }
                    )
        return vehicles


class PlateDetector:
    def __init__(self, model_path: str = "model/plate_model.pt"):
        self.model_path = Path(model_path)
        self.model = None
        if self.model_path.exists():
            self.model = YOLO(str(self.model_path))

    def detect(self, image):
        if self.model is not None:
            return self._detect_yolo(image)
        return self._detect_opencv(image)

    def _detect_yolo(self, image):
        results = self.model(image, verbose=False)
        plates = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                plates.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(float(box.conf[0]), 4),
                    }
                )
        return plates

    def _detect_opencv(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)
        dilated = cv2.dilate(edged, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        img_h, img_w = image.shape[:2]
        plates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 60 or h < 16:
                continue
            aspect = w / float(h)
            area_ratio = (w * h) / float(img_h * img_w)
            if 2.0 <= aspect <= 6.0 and 0.002 <= area_ratio <= 0.2:
                plates.append({"bbox": [x, y, x + w, y + h], "confidence": 1.0})
        return plates
