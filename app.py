import base64
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from pipeline import PlatePipeline

st.set_page_config(page_title="Vall — Deteksi Plat Nomor", page_icon="🚗", layout="wide")

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
    radial-gradient(900px 460px at 85% -10%, rgba(255,176,32,0.09), transparent 60%),
    linear-gradient(180deg, #101418 0%, #0C0F12 100%);
}

.block-container {
  padding-top: 2.4rem;
  padding-bottom: 4rem;
  max-width: 1120px;
}

h1, h2, h3 {
  font-family: 'Space Grotesk', sans-serif;
  color: var(--text);
  letter-spacing: -0.02em;
}
h1 {
  font-size: 2.6rem;
  line-height: 1.04;
  font-weight: 700;
}

/* clean top bar */
[data-testid="stHeader"] {
  background: transparent !important;
  border-bottom: 1px solid rgba(255,255,255,0.04);
}
[data-testid="stToolbar"],
[data-testid="stDeployButton"] {
  display: none !important;
}
[data-testid="stSidebarCollapsedControl"] {
  color: var(--text);
}

/* sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #151A1F, #101418);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stRadio label {
  color: var(--muted);
  font-weight: 500;
}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
  color: var(--accent);
}

/* tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.4rem;
  background: transparent;
  margin-top: 0.7rem;
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
[data-baseweb="tab-panel"] {
  padding-top: 1.4rem;
}

/* buttons */
.stButton > button {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 500;
  color: #12171A;
  background: var(--accent);
  border: 1px solid var(--accent);
  border-radius: 999px;
  padding: 0.6rem 1.5rem;
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
.stButton > button[kind="secondary"] {
  background: transparent;
  color: var(--text);
  border-color: var(--border);
}
.stButton > button[kind="secondary"]:hover {
  border-color: var(--accent);
  color: var(--accent);
  box-shadow: none;
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

/* ---------- landing ---------- */
.lp-hero {
  padding: 1.2rem 0 2rem;
}
.lp-tag {
  color: var(--accent);
  font-weight: 600;
  font-size: 1rem;
  margin-bottom: .8rem;
}
.lp-title {
  font-size: 3.1rem;
  line-height: 1.02;
  max-width: 14ch;
}
.lp-sub {
  color: var(--muted);
  font-size: 1.12rem;
  line-height: 1.65;
  max-width: 42ch;
  margin-top: 1.2rem;
}
.lp-cta {
  display: flex;
  gap: .7rem;
  margin-top: 1.8rem;
  flex-wrap: wrap;
}
.lp-media {
  position: relative;
  border: 1px solid var(--border);
  border-radius: 18px;
  overflow: hidden;
  background: var(--surface);
  box-shadow: 0 24px 60px rgba(0,0,0,0.35);
}
.lp-media img.main {
  display: block;
  width: 100%;
  height: auto;
}
.lp-media .logo-badge {
  position: absolute;
  top: 14px;
  left: 14px;
  width: 92px;
  height: auto;
  border-radius: 12px;
  border: 1px solid var(--border);
  background: #101418;
  padding: 6px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.45);
}
.lp-detail {
  display: grid;
  grid-template-columns: 3fr 2fr;
  gap: 2.5rem;
  padding: 0.4rem 0 1rem;
}
.lp-detail p {
  color: var(--muted);
  line-height: 1.7;
  margin: 0;
}
.lp-list {
  list-style: none;
  margin: 0;
  padding: 0;
}
.lp-list li {
  color: var(--text);
  padding: 0.55rem 0;
  border-bottom: 1px solid var(--border);
  font-size: 0.97rem;
}
.lp-list li:last-child {
  border-bottom: none;
}
.lp-list .k {
  color: var(--accent);
  font-weight: 600;
}
@media (max-width: 768px) {
  .lp-detail {
    grid-template-columns: 1fr;
    gap: 1.4rem;
  }
}

.lp-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 1.3rem 3rem;
  padding: 1.6rem 0;
  margin: 2rem 0 0.6rem;
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
}
.lp-item .t {
  color: var(--text);
  font-weight: 600;
  font-size: 1.02rem;
}
.lp-item .d {
  color: var(--muted);
  font-size: .92rem;
  margin-top: .15rem;
}

.lp-note {
  color: var(--muted);
  font-size: .9rem;
  text-align: center;
  margin-top: 2.6rem;
}

.divider {
  border: none;
  border-top: 1px solid var(--border);
  margin: 2.6rem 0;
}

/* ---------- tool ---------- */
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

@media (max-width: 768px) {
  [data-testid="stSidebarCollapsedControl"] {
    width: 44px;
    height: 44px;
  }
  .block-container {
    padding-top: 4.4rem;
    padding-left: 1rem;
    padding-right: 1rem;
  }
  .lp-title {
    font-size: 2.3rem;
  }
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


def _b64_image(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def go_deteksi():
    st.session_state["nav"] = "Deteksi"


def go_demo():
    st.session_state["nav"] = "Deteksi"
    st.session_state["demo_scroll"] = True


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
    for r in results:
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
        st.image("LogoVall.jpeg", width=150)
    except Exception:
        pass
    route = st.radio("Navigasi", ["Beranda", "Deteksi"], label_visibility="collapsed", key="nav")
    st.divider()
    st.caption("Membaca plat nomor dari foto kendaraan.")

# ---------------- Landing ----------------
if route == "Beranda":
    st.markdown(
        "<div class='lp-hero'>"
        "<div class='lp-tag'>Pendeteksi plat nomor</div>"
        "<div class='lp-title'>Plat nomor terbaca dalam hitungan detik.</div>"
        "<div class='lp-sub'>Unggah foto kendaraan atau pakai kamera langsung dari HP. "
        "Sistem mendeteksi dan membaca plat nomor secara otomatis.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    hero_l, hero_r = st.columns([6, 5], gap="large")
    with hero_l:
        st.markdown("<div class='lp-cta'>", unsafe_allow_html=True)
        st.button("Mulai deteksi", type="primary", use_container_width=True, on_click=go_deteksi)
        st.button("Lihat contoh", type="secondary", use_container_width=True, on_click=go_demo)
        st.markdown("</div>", unsafe_allow_html=True)
    with hero_r:
        st.markdown(
            "<div class='lp-media'><img class='main' src='data:image/jpeg;base64,"
            + _b64_image("samples/sample_plate1.jpg")
            + "' alt='contoh kendaraan'>"
            "<img class='logo-badge' src='data:image/jpeg;base64,"
            + _b64_image("LogoVall.jpeg")
            + "' alt='Vall'></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='lp-strip'>"
        "<div class='lp-item'><div class='t'>Foto apa pun</div>"
        "<div class='d'>JPG, PNG, sampai foto webcam.</div></div>"
        "<div class='lp-item'><div class='t'>Hasil seketika</div>"
        "<div class='d'>Teks plat ditandai pada gambar.</div></div>"
        "<div class='lp-item'><div class='t'>Gratis & online</div>"
        "<div class='d'>Jalan langsung di browser.</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("## Detail proyek")
    detail_l, detail_r = st.columns([3, 2], gap="large")
    with detail_l:
        st.markdown(
            "<div class='lp-detail'><p>"
            "Aplikasi web ini membaca plat nomor dari foto kendaraan menggunakan "
            "optical character recognition (OCR). Unggah gambar atau tangkap lewat "
            "kamera HP, lalu sistem menandai lokasi teks plat pada gambar. "
            "Dirancang ringan agar bisa dipakai langsung dari perangkat apa pun."
            "</p></div>",
            unsafe_allow_html=True,
        )
    with detail_r:
        st.markdown(
            "<ul class='lp-list'>"
            "<li><span class='k'>Fungsi</span> — deteksi & baca teks plat</li>"
            "<li><span class='k'>Teknologi</span> — EasyOCR, Python, Streamlit</li>"
            "<li><span class='k'>Deploy</span> — Streamlit Cloud, gratis</li>"
            "<li><span class='k'>Catatan</span> — akurasi bergantung kualitas foto</li>"
            "</ul>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='lp-note'>Vall · dibuat dengan Streamlit</div>", unsafe_allow_html=True)

# ---------------- Tool ----------------
else:
    if st.session_state.get("demo_scroll"):
        st.session_state["demo_scroll"] = False

    st.markdown(
        "<div class='demo-head'>"
        "<h2 style='margin-bottom:0'>Coba dengan contoh</h2>"
        "<p>Belum punya foto? Pilih salah satu contoh di bawah.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    demo_cols = st.columns(2)
    for col, (path, name, key) in zip(
        demo_cols,
        [
            ("samples/sample_plate1.jpg", "Contoh 1", "demo1"),
            ("samples/sample_plate2.jpg", "Contoh 2", "demo2"),
        ],
    ):
        with col:
            st.markdown("<div class='round'>", unsafe_allow_html=True)
            st.image(path, caption=name, use_container_width=True)
            if st.button("Uji contoh ini", key=key):
                img = Image.open(path).convert("RGB")
                process_and_render(img, f"Hasil — {name}")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
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
