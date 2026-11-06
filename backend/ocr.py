import re

import easyocr


ALLOWED_CHARS = re.compile(r"[A-Z0-9]")
PLATE_PATTERN = re.compile(r"(?=.*[A-Z])(?=.*[0-9])")


class PlateOCR:
    def __init__(self, languages=("en",)):
        self.reader = easyocr.Reader(languages, gpu=False)

    def read(self, plate_crop):
        result = self.reader.readtext(plate_crop, detail=1, allowlist=ALLOWED_CHARS.pattern)
        if not result:
            return None, 0.0
        best = max(result, key=lambda r: r[2])
        text = best[1].replace(" ", "")
        confidence = best[2]
        if not PLATE_PATTERN.search(text):
            return None, confidence
        return text, confidence
