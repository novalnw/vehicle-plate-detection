import cv2

from detector import PlateDetector, VehicleDetector
from ocr import PlateOCR


class PlatePipeline:
    def __init__(self, vehicle_model="yolov8n.pt", plate_model="model/plate_model.pt"):
        self.vehicle_detector = VehicleDetector(vehicle_model)
        self.plate_detector = PlateDetector(plate_model)
        self.ocr = PlateOCR()

    def process_image_bytes(self, image):
        return self.process_image(image)

    def process_image(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        vehicles = self.vehicle_detector.detect(image)
        results = []
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle["bbox"]
            crop = image[y1:y2, x1:x2]
            try:
                plates = self.plate_detector.detect(crop)
            except FileNotFoundError:
                plates = []

            vehicle_plate = None
            for plate in plates:
                px1, py1, px2, py2 = plate["bbox"]
                plate_crop = crop[py1:py2, px1:px2]
                text, conf = self.ocr.read(plate_crop)
                if text is not None:
                    vehicle_plate = {"text": text, "confidence": round(float(conf), 4)}
                    break

            results.append(
                {
                    "vehicle": vehicle["label"],
                    "vehicle_confidence": vehicle["confidence"],
                    "vehicle_bbox": [x1, y1, x2, y2],
                    "plate": vehicle_plate,
                }
            )
        return results
