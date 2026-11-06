import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from pipeline import PlatePipeline

st.set_page_config(page_title="Deteksi Plat Nomor", page_icon="🚗", layout="wide")

with st.sidebar:
    try:
        st.image("LogoVall.jpeg", use_container_width=True)
    except Exception:
        pass
    st.title("Deteksi Plat Nomor")
    st.caption("Deteksi plat nomor menggunakan EasyOCR.")
    st.divider()
    st.markdown(
        "**Cara pakai:**\n"
        "- Tab **Upload Gambar**: unggah foto kendaraan.\n"
        "- Tab **Realtime Webcam**: ambil foto dari kamera.\n\n"
        "Hasil plat nomor akan dianotasi pada gambar."
    )


def get_pipeline():
    if "pipeline" not in st.session_state:
        st.session_state["pipeline"] = PlatePipeline()
    return st.session_state["pipeline"]


def draw_annotations(img, results):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    for r in results:
        x1, y1, x2, y2 = r["vehicle_bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=4)
        label = f"{r['vehicle']} {r['vehicle_confidence']:.2f}"
        draw.text((x1, y1 - 34), label, fill="lime", font=font)
        if r["plate"]:
            px1, py1, px2, py2 = r["plate"]["bbox"]
            draw.rectangle([px1, py1, px2, py2], outline="red", width=4)
            draw.text((px1, py1 - 34), r["plate"]["text"], fill="red", font=font)
    return img


def process_and_render(pil_image, caption):
    pipeline = get_pipeline()
    arr = np.array(pil_image)
    results = pipeline.process_image_bytes(arr)
    annotated = draw_annotations(pil_image.copy(), results)
    st.image(annotated, caption=caption, use_container_width=True)

    if not results:
        st.warning("Tidak ada plat nomor terdeteksi.")
        return

    st.subheader("Hasil Deteksi")
    for idx, r in enumerate(results, 1):
        with st.container(border=True):
            plate_text = r["plate"]["text"] if r["plate"] else "-"
            plate_conf = r["plate"]["confidence"] if r["plate"] else "-"
            st.markdown(f"**Hasil #{idx}:** `{plate_text}`  (confidence {plate_conf})")


st.title("🚗 Deteksi Plat Nomor")
st.caption("Deteksi plat nomor dengan EasyOCR")

tab_upload, tab_webcam = st.tabs(["📤 Upload Gambar", "📷 Realtime Webcam"])

with tab_upload:
    st.header("Upload Gambar")
    uploaded = st.file_uploader("Pilih gambar kendaraan", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is not None:
        image = Image.open(BytesIO(uploaded.read())).convert("RGB")
        st.image(image, caption="Gambar asli", use_container_width=True)
        if st.button("🔍 Deteksi", key="detect_upload"):
            process_and_render(image, "Hasil deteksi")

with tab_webcam:
    st.header("Realtime Webcam")
    st.caption("Aktifkan kamera, ambil foto, lalu deteksi.")
    pic = st.camera_input("Ambil foto dari webcam")
    if pic is not None:
        image = Image.open(BytesIO(pic.read())).convert("RGB")
        st.image(image, caption="Foto dari webcam", use_container_width=True)
        if st.button("🔍 Deteksi", key="detect_webcam"):
            process_and_render(image, "Hasil deteksi webcam")
