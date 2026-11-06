# Vehicle Plate Detection

Deteksi kendaraan + baca plat nomor (Indonesia) menggunakan YOLOv8 dan EasyOCR.

## Fitur

- Deteksi kendaraan (mobil, motor, bus, truk) — YOLOv8 pretrained
- Deteksi plat nomor — YOLOv8 (dilatih di dataset plat Indonesia)
- OCR plat nomor — EasyOCR
- API REST — FastAPI
- Web app — Streamlit (upload gambar + real-time webcam)

## Jalankan Web App (Streamlit)

```bash
./venv/bin/python -m streamlit run app.py
```

Buka `http://localhost:8501`.

### Deploy ke Streamlit Cloud (gratis)

1. Buat repo di GitHub dan push folder ini (tanpa `venv/`, `dataset/`, `model/`).
2. Buka https://share.streamlit.io → **Create app** → pilih repo tsb.
3. Main file: `app.py`, Python version: 3.10+.
4. App langsung online di URL gratis, bisa dibuka dari HP (kamera HP bisa dipakai).

## Struktur

```
vehicle-plate-detection/
├── backend/
│   ├── main.py        # FastAPI app
│   ├── pipeline.py    # orchestration: detect -> crop -> ocr
│   ├── detector.py    # YOLO vehicle & plate detection
│   └── ocr.py         # EasyOCR reader
├── scripts/
│   └── train_plate.py # training deteksi plat
├── dataset/
│   └── indonesian-plate-yolo/  # 561 gambar plat Indonesia (YOLO format)
├── model/             # hasil training (plate_model.pt)
├── samples/           # gambar test
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Training deteksi plat

```bash
./venv/bin/python scripts/train_plate.py --epochs 50
```

Hasil disimpan di `runs/detect/plate_model/weights/best.pt`. Salin ke `model/plate_model.pt`:

```bash
cp runs/detect/plate_model/weights/best.pt model/plate_model.pt
```

## Jalankan API

```bash
./venv/bin/python backend/main.py
```

Coba:

```bash
curl -F "file=@samples/<foto>.jpg" http://localhost:8000/detect
```

## Dataset

- **Plat Indonesia (deteksi):** [Indonesian License Plate Lighting Dataset](https://github.com/salsabilaputri95/Indonesian-License-Plate-Lighting-Dataset) — 561 gambar, format YOLOv8, kelas `number-plate`, lisensi CC BY 4.0.
- **Plat Indonesia (OCR):** [Kaggle](https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset)

## Catatan CPU-only

Tanpa GPU, training & inference lebih lambat. Gunakan `yolov8n.pt` (nano) untuk kecepatan.
