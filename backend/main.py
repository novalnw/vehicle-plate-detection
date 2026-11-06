import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from pipeline import PlatePipeline

app = FastAPI(title="Vehicle Plate Detection API")

pipeline = PlatePipeline()


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = pipeline.process_image(tmp_path)
        return JSONResponse(content={"vehicles": result})
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
