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


def build_css(dark):
    if dark:
        paper, surface, surface2, border_, ink, muted = (
            "#0F1316", "#171C20", "#1E2429", "#2A333A", "#E6EDF2", "#8F9CA8",
        )
        accent, accent_soft, ok, danger = (
            "#5B9BFF", "#162B44", "#36C48E", "#FF6B6B",
        )
        btn_text, btn_hover = "#0B1220", "#7FB4FF"
        shadow = "rgba(0,0,0,0.5)"
    else:
        paper, surface, surface2, border_, ink, muted = (
            "#F3F5F6", "#FFFFFF", "#ECF0F2", "#D8DEE3", "#15181B", "#5B6670",
        )
        accent, accent_soft, ok, danger = (
            "#17488F", "#E8F0FA", "#157A4E", "#BC3A2B",
        )
        btn_text, btn_hover = "#FFFFFF", "#103A6E"
        shadow = "rgba(21,24,27,0.08)"

    root = (
        ":root{"
        f"--paper:{paper};--surface:{surface};--surface2:{surface2};"
        f"--border:{border_};--ink:{ink};--muted:{muted};"
        f"--accent:{accent};--accent-soft:{accent_soft};"
        f"--ok:{ok};--danger:{danger};--shadow:{shadow};"
        f"--btn-text:{btn_text};--btn-hover:{btn_hover};"
        "}"
    )
    body = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; color: var(--ink); }
.stApp { background: linear-gradient(180deg, var(--paper) 0%, var(--surface2) 100%); }
.block-container { padding-top: 2.2rem; padding-bottom: 4.5rem; max-width: 1120px; }
h1, h2, h3 { font-family: 'Source Serif 4', Georgia, serif; color: var(--ink); letter-spacing: -0.01em; }
h1 { font-size: 2.75rem; line-height: 1.08; font-weight: 700; }
h2 { font-size: 1.9rem; font-weight: 700; }
h3 { font-size: 1.3rem; font-weight: 600; }
[data-testid="stHeader"] { background: transparent !important; border-bottom: 1px solid var(--border); }
[data-testid="stToolbar"], [data-testid="stDeployButton"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { color: var(--ink); }
section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] .stRadio label { color: var(--muted); font-weight: 500; }
section[data-testid="stSidebar"] .stRadio label:has(input:checked) { color: var(--accent); }
.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; background: transparent; margin-top: 0.6rem; }
.stTabs [data-baseweb="tab"] { height: auto; padding: 0.55rem 1.1rem; background: var(--surface); border: 1px solid var(--border); border-radius: 7px; color: var(--muted); font-weight: 500; }
.stTabs [aria-selected="true"] { background: var(--accent) !important; color: var(--btn-text) !important; border-color: var(--accent) !important; }
[data-baseweb="tab-panel"] { padding-top: 1.3rem; }
.stButton > button { font-family: 'Inter', sans-serif; font-weight: 600; color: var(--btn-text); background: var(--accent); border: 1px solid var(--accent); border-radius: 8px; padding: 0.55rem 1.35rem; transition: background .12s ease, transform .12s ease; }
.stButton > button:hover { background: var(--btn-hover); border-color: var(--btn-hover); transform: translateY(-1px); }
.stButton > button:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
.stButton > button[kind="secondary"] { background: transparent; color: var(--ink); border-color: var(--border); }
.stButton > button[kind="secondary"]:hover { border-color: var(--ink); background: transparent; transform: none; }
.stFileUploader section { background: var(--surface); border: 1px dashed var(--border); border-radius: 10px; }
.stFileUploader section:hover { border-color: var(--accent); }
.lp-hero { padding: 1.6rem 0 2rem; border-bottom: 3px double var(--ink); }
.lp-tag { color: var(--accent); font-weight: 600; font-size: 0.95rem; margin-bottom: .9rem; }
.lp-title { font-family: 'Source Serif 4', Georgia, serif; font-size: 3rem; line-height: 1.05; font-weight: 700; max-width: 15ch; }
.lp-sub { color: var(--muted); font-size: 1.1rem; line-height: 1.65; max-width: 44ch; margin-top: 1.1rem; }
.lp-cta { display: flex; gap: .7rem; margin-top: 1.8rem; flex-wrap: wrap; }
.lp-media { background: var(--surface); border: 1.5px solid var(--ink); border-radius: 8px; padding: 8px; box-shadow: 12px 14px 0 var(--shadow); }
.lp-media img.main { display: block; width: 100%; height: auto; border-radius: 4px; }
.lp-strip { display: flex; flex-wrap: wrap; gap: 1.4rem 3rem; padding: 1.7rem 0; margin: 2.4rem 0 1rem; border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); }
.lp-item .t { color: var(--ink); font-weight: 600; font-size: 1.02rem; }
.lp-item .d { color: var(--muted); font-size: .92rem; margin-top: .15rem; }
.lp-note { color: var(--muted); font-size: .9rem; text-align: center; margin-top: 2.8rem; }
.lp-detail p { color: var(--muted); line-height: 1.75; margin: 0; }
.lp-list { list-style: none; margin: 0; padding: 0; }
.lp-list li { color: var(--ink); padding: 0.55rem 0; border-bottom: 1px solid var(--border); font-size: 0.97rem; }
.lp-list li:last-child { border-bottom: none; }
.lp-list .k { color: var(--accent); font-weight: 600; }
.divider { border: none; border-top: 1px solid var(--border); margin: 2.8rem 0; }
.demo-head { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; margin-bottom: .5rem; }
.demo-head p { color: var(--muted); margin: 0; }
.round { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: .9rem; }
.result-block { background: var(--surface); border: 2.5px solid var(--ink); border-radius: 8px; padding: 0.9rem 1.2rem; margin-top: .6rem; box-shadow: 6px 7px 0 var(--shadow); }
.result-plate { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; color: var(--ink); letter-spacing: .05em; }
.result-meta { color: var(--muted); font-size: .9rem; margin-top: .15rem; }
.stWarning { background: var(--accent-soft); color: var(--accent); border-radius: 8px; border: 1px solid transparent; }
@media (max-width: 768px) {
  [data-testid="stSidebarCollapsedControl"] { width: 44px; height: 44px; }
  .block-container { padding-top: 4.2rem; padding-left: 1rem; padding-right: 1rem; }
  .lp-title { font-size: 2.3rem; }
}
@media (prefers-reduced-motion: reduce) { .stButton > button { transition: none; } }
"""
    return "<style>" + root + body + "</style>"


if "dark" not in st.session_state:
    st.session_state["dark"] = True
st.markdown(build_css(st.session_state["dark"]), unsafe_allow_html=True)


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
        draw.rectangle([x1, y1, x2, y2], outline=(91, 155, 255), width=4)
        label = f"{r['vehicle']} · {r['vehicle_confidence']:.2f}"
        draw.text((x1, y1 - 34), label, fill=(91, 155, 255), font=font_small)
        if r["plate"]:
            px1, py1, px2, py2 = r["plate"]["bbox"]
            draw.rectangle([px1, py1, px2, py2], outline=(54, 196, 142), width=4)
            draw.text((px1, py1 - 34), r["plate"]["text"], fill=(54, 196, 142), font=font)
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
    st.markdown("### Vall Plate Detector")
    dark = st.toggle("Mode gelap", key="dark")
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
            + "' alt='contoh kendaraan'></div>",
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

    st.markdown(
        "<p style='color:var(--muted);line-height:1.75;max-width:70ch'>"
        "Vall Plate Detector adalah proyek web yang membaca plat nomor dari foto "
        "kendaraan. Aplikasi ini memakai optical character recognition (OCR) untuk "
        "menemukan dan menandai teks plat pada gambar, sehingga hasilnya langsung "
        "terlihat tanpa perlu disetel manual."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown("### Cara kerja")
    st.markdown(
        "<ol class='lp-list' style='list-style:decimal inside'>"
        "<li><span class='k'>Pasang gambar</span> — unggah foto atau ambil via kamera HP.</li>"
        "<li><span class='k'>Pindai</span> — sistem memindai gambar untuk mencari teks.</li>"
        "<li><span class='k'>Filter</span> — teks yang cocok dengan pola plat disaring.</li>"
        "<li><span class='k'>Tandai</span> — lokasi dan teks plat ditandai pada gambar.</li>"
        "</ol>",
        unsafe_allow_html=True,
    )

    detail_l, detail_r = st.columns([3, 2], gap="large")
    with detail_l:
        st.markdown(
            "<div class='lp-detail'><p>"
            "Proyek ini menggunakan EasyOCR, pustaka Python untuk pengenalan teks "
            "pada gambar, dan berjalan di atas Streamlit sebagai antarmuka web. "
            "Seluruh proses berlangsung otomatis: foto masuk, teks plat keluar. "
            "Aplikasi tetap ringan karena hanya aktif saat kamu menekan tombol "
            "deteksi."
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
