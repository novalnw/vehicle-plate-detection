import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from pipeline import PlatePipeline

st.set_page_config(page_title="Deteksi Plat Nomor", page_icon="🚗", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

:root {
  --bg: #101418;
  --surface: #171D22;
  --surface2: #1E262D;
  --border: #2A333B;
  --text: #E8EEF4;
  --muted: #93A1AE;
  --accent: #FFB020;
  --ok: #35C48E;
  --danger: #FF6B6B;
}

html, body, [class*="css"] {
  font-family: 'Inter', system-ui, sans-serif;
}

.stApp {
  background:
    radial-gradient(900px 420px at 85% -10%, rgba(255,176,32,0.08), transparent 60%),
    linear-gradient(180deg, #101418 0%, #0C0F12 100%);
}

.block-container {
  padding-top: 2.6rem;
  padding-bottom: 4rem;
  max-width: 1100px;
}

/* headers */
h1, h2, h3 {
  font-family: 'Space Grotesk', sans-serif;
  color: var(--text);
  letter-spacing: -0.02em;
}
h1 {
  font-size: 2.5rem;
  line-height: 1.05;
  font-weight: 700;
}

/* sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #151A1F, #101418);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * {
  font-family: 'Inter', sans-serif;
}
section[data-testid="stSidebar"] h1 {
  font-size: 1.5rem;
  color: var(--text);
}

/* tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.4rem;
  background: transparent;
}
.stTabs [data-baseweb="tab"] {
  height: auto;
  padding: 0.65rem 1.2rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 999px;
  color: var(--muted);
  font-weight: 500;
}
.stTabs [aria-selected="true"] {
  background: var(--accent) !important;
  color: #12171A !important;
  border-color: var(--accent) !important;
}

/* buttons */
.stButton > button {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 500;
  color: #12171A;
  background: var(--accent);
  border: 1px solid var(--accent);
  border-radius: 999px;
  padding: 0.55rem 1.4rem;
  transition: transform .12s ease, box-shadow .12s ease;
}
.stButton > button:hover {
  background: #FFC44D;
  border-color: #FFC44D;
  transform: translateY(-1px);
  box-shadow: 0 8px 24px rgba(255,176,32,0.25);
}
.stButton > button:focus-visible {
  outline: 2px solid #fff;
  outline-offset: 2px;
}

/* file uploader */
.stFileUploader section {
  background: var(--surface);
  border: 1px dashed var(--border);
  border-radius: 14px;
}
.stFileUploader section:hover {
  border-color: var(--accent);
}

/* hero */
.hero {
  border-bottom: 1px solid var(--border);
  padding-bottom: 1.8rem;
  margin-bottom: 2rem;
}
.hero .tag {
  color: var(--accent);
  font-weight: 600;
  font-size: .95rem;
  margin-bottom: .6rem;
}
.hero p {
  color: var(--muted);
  font-size: 1.05rem;
  max-width: 42ch;
  line-height: 1.6;
}

/* feature strip */
.feature-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 1.2rem 2.5rem;
  padding: 1.1rem 0 1.6rem;
  margin-bottom: 2rem;
  border-bottom: 1px solid var(--border);
}
.feature {
  display: flex;
  align-items: baseline;
  gap: .7rem;
  color: var(--text);
}
.feature .step {
  font-family: 'Space Grotesk', sans-serif;
  color: var(--accent);
  font-weight: 700;
  font-size: 1.15rem;
}
.feature .t {
  color: var(--text);
  font-weight: 600;
}
.feature .d {
  color: var(--muted);
  font-size: .92rem;
}

/* demo */
.demo-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: .4rem;
}
.demo-head p {
  color: var(--muted);
  margin: 0;
}
.round {
  border: 1px solid var(--border);
  border-radius: 16px;
  background: var(--surface);
  padding: .9rem;
}

/* results */
.result-block {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1rem 1.2rem;
  margin-top: .6rem;
}
.result-plate {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: .04em;
}
.result-meta {
  color: var(--muted);
  font-size: .9rem;
}
.stWarning {
  background: rgba(255,176,32,0.1);
  color: var(--accent);
  border-radius: 12px;
}

@media (prefers-reduced-motion: reduce) {
  .stButton > button { transition: none; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_pipeline():
    if "pipeline" not in st.session_state:
        st.session_state["pipeline"] = PlatePipeline()
    return st.session_state["pipeline"]


def draw_annotations(img, results):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
        font_small = font
    for r in results:
        x1, y1, x2, y2 = r["vehicle_bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 176, 32), width=4)
        label = f"{r['vehicle']} · {r['vehicle_confidence']:.2f}"
        draw.text((x1, y1 - 34), label, fill=(255, 176, 32), font=font_small)
        if r["plate"]:
            px1, py1, px2, py2 = r["plate"]["bbox"]
            draw.rectangle([px1, py1, px2, py2], outline=(53, 196, 142), width=4)
            draw.text((px1, py1 - 34), r["plate"]["text"], fill=(53, 196, 142), font=font)
    return img


def render_results(results):
    if not results:
        st.warning("Tidak ada plat nomor yang terdeteksi pada gambar.")
        return
    for idx, r in enumerate(results, 1):
        plate_text = r["plate"]["text"] if r["plate"] else "-"
        plate_conf = r["plate"]["confidence"] if r["plate"] else "-"
        st.markdown(
            f"<div class='result-block'><span class='result-plate'>{plate_text}</span>"
            f"<div class='result-meta'>confidence {plate_conf}</div></div>",
            unsafe_allow_html=True,
        )


def process_and_render(pil_image, caption):
    with st.spinner("Membaca plat nomor…"):
        pipeline = get_pipeline()
        arr = np.array(pil_image)
        results = pipeline.process_image_bytes(arr)
        annotated = draw_annotations(pil_image.copy(), results)
    st.image(annotated, caption=caption, use_container_width=True)
    st.markdown("### Hasil deteksi")
    render_results(results)


# ---------------- Sidebar ----------------
with st.sidebar:
    try:
        st.image("LogoVall.jpeg", use_container_width=True)
    except Exception:
        pass
    st.markdown("## Deteksi Plat Nomor")
    st.caption("Membaca plat nomor dari foto kendaraan.")
    st.divider()
    st.markdown("Ditenagai EasyOCR · berjalan di Streamlit Cloud.")

# ---------------- Hero ----------------
st.markdown(
    "<div class='hero'>"
    "<div class='tag'>Pendeteksi plat nomor</div>"
    "<h1>Foto kendaraan, langsung terbaca platnya.</h1>"
    "<p>Unggah foto atau pakai kamera. Sistem menemukan teks plat nomor "
    "dan menandainya pada gambar.</p>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='feature-strip'>"
    "<div class='feature'><span class='step'>1</span><span>"
    "<span class='t'>Unggah gambar</span> · <span class='d'>kirim foto kendaraan</span></span></div>"
    "<div class='feature'><span class='step'>2</span><span>"
    "<span class='t'>Webcam langsung</span> · <span class='d'>ambil dari kamera</span></span></div>"
    "<div class='feature'><span class='step'>3</span><span>"
    "<span class='t'>Hasil anotasi</span> · <span class='d'>plat ditandai di gambar</span></span></div>"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------- Demo (contoh) ----------------
st.markdown(
    "<div class='demo-head'>"
    "<h2 style='margin-bottom:0'>Coba dengan contoh</h2>"
    "<p>Belum punya foto? Pilih salah satu contoh di bawah.</p>"
    "</div>",
    unsafe_allow_html=True,
)

demo_cols = st.columns(2)
samples = [
    ("samples/sample_plate1.jpg", "Contoh 1", "demo1"),
    ("samples/sample_plate2.jpg", "Contoh 2", "demo2"),
]
for col, (path, name, key) in zip(demo_cols, samples):
    with col:
        with st.container():
            st.markdown("<div class='round'>", unsafe_allow_html=True)
            st.image(path, caption=name, use_container_width=True)
            if st.button("Uji contoh ini", key=key):
                img = Image.open(path).convert("RGB")
                process_and_render(img, f"Hasil — {name}")
            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr style='border-color:var(--border);margin:2.6rem 0'>", unsafe_allow_html=True)

# ---------------- Fitur utama ----------------
st.markdown("## Deteksi langsung")

tab_upload, tab_webcam = st.tabs(["Unggah gambar", "Webcam langsung"])

with tab_upload:
    uploaded = st.file_uploader("Pilih foto kendaraan", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is not None:
        image = Image.open(BytesIO(uploaded.read())).convert("RGB")
        st.image(image, caption="Gambar asli", use_container_width=True)
        if st.button("Deteksi dari gambar", key="detect_upload"):
            process_and_render(image, "Hasil deteksi")

with tab_webcam:
    st.caption("Aktifkan kamera, ambil foto, lalu deteksi.")
    pic = st.camera_input("Ambil foto dari webcam")
    if pic is not None:
        image = Image.open(BytesIO(pic.read())).convert("RGB")
        st.image(image, caption="Foto dari webcam", use_container_width=True)
        if st.button("Deteksi dari webcam", key="detect_webcam"):
            process_and_render(image, "Hasil deteksi webcam")
