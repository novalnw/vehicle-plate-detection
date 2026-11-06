import re

from ocr import PlateOCR

PLATE_RE = re.compile(r"^[A-Z]{1,3}\d{1,4}[A-Z]{0,3}\d{0,4}$")


class PlatePipeline:
    def __init__(self):
        self.ocr = PlateOCR()

    def process_image_bytes(self, image):
        return self.process_image(image)

    def process_image(self, image):
        scans = self.ocr.scan(image)
        results = []
        for item in scans:
            if not PLATE_RE.match(item["text"]):
                continue
            xs = [p[0] for p in item["bbox"]]
            ys = [p[1] for p in item["bbox"]]
            px1, py1 = int(min(xs)), int(min(ys))
            px2, py2 = int(max(xs)), int(max(ys))
            results.append(
                {
                    "vehicle": "kendaraan",
                    "vehicle_confidence": round(float(item["confidence"]), 4),
                    "vehicle_bbox": [px1, py1, px2, py2],
                    "plate": {
                        "text": item["text"],
                        "confidence": round(float(item["confidence"]), 4),
                        "bbox": [px1, py1, px2, py2],
                    },
                }
            )
        return results
