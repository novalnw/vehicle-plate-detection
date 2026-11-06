import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from pipeline import PlatePipeline

st.set_page_config(page_title="Deteksi Plat Nomor", page_icon="🚗", layout="wide")


def get_pipeline():
    if "pipeline" not in st.session_state:
        st.session_state["pipeline"] = PlatePipeline()
    return st.session_state["pipeline"]


@st.cache_resource
def get_ocr_ready():
    sys.path.insert(0, str(Path(__file__).parent / "backend"))
    from ocr import PlateOCR

    return PlateOCR()


def draw_annotations(image, results):
    img = image.copy()
    for r in results:
        x1, y1, x2, y2 = r["vehicle_bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{r['vehicle']} {r['vehicle_confidence']:.2f}"
        cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if r["plate"]:
            px1, py1, px2, py2 = r["plate"]["bbox"]
            cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 2)
            cv2.putText(
                img,
                r["plate"]["text"],
                (px1, py1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
    return img


def process_and_render(image, caption):
    pipeline = get_pipeline()
    results = pipeline.process_image_bytes(image)
    annotated = draw_annotations(image, results)
    st.image(annotated, channels="BGR", caption=caption, use_container_width=True)

    if not results:
        st.warning("Tidak ada kendaraan terdeteksi.")
        return

    st.subheader("Hasil Deteksi")
    for idx, r in enumerate(results, 1):
        with st.container(border=True):
            plate_text = r["plate"]["text"] if r["plate"] else "-"
            plate_conf = r["plate"]["confidence"] if r["plate"] else "-"
            st.markdown(f"**Kendaraan #{idx}:** {r['vehicle']} (confidence {r['vehicle_confidence']:.2f})")
            st.markdown(f"- **Plat nomor:** `{plate_text}`")
            st.markdown(f"- **Confidence OCR:** {plate_conf}")


st.title("🚗 Deteksi Plat Nomor")
st.caption("Deteksi kendaraan + OCR plat nomor dengan YOLOv8 & EasyOCR")

tab_upload, tab_webcam = st.tabs(["📤 Upload Gambar", "📷 Realtime Webcam"])

with tab_upload:
    st.header("Upload Gambar")
    uploaded = st.file_uploader("Pilih gambar kendaraan", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, channels="BGR", caption="Gambar asli", use_container_width=True)
        if st.button("🔍 Deteksi", key="detect_upload"):
            process_and_render(image, "Hasil deteksi")

with tab_webcam:
    st.header("Realtime Webcam")
    st.caption("Aktifkan kamera, ambil foto, lalu deteksi.")
    pic = st.camera_input("Ambil foto dari webcam")
    if pic is not None:
        file_bytes = np.frombuffer(pic.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, channels="BGR", caption="Foto dari webcam", use_container_width=True)
        if st.button("🔍 Deteksi", key="detect_webcam"):
            process_and_render(image, "Hasil deteksi webcam")
