import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# --------------------------- Page configuration ---------------------------
st.set_page_config(
    page_title="ClearVision AI | Vision Restoration & Detection",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- Styling ---------------------------
st.markdown("""
<style>
/* ClearVision AI — high-contrast glassmorphism theme */
:root {
  color-scheme: dark;
  --cv-bg: #07111f;
  --cv-panel: rgba(17, 34, 57, .76);
  --cv-line: rgba(148, 190, 226, .24);
  --cv-text: #f4f8ff;
  --cv-muted: #c2d2e5;
  --cv-cyan: #67e8f9;
  --cv-mint: #6ee7b7;
}
html, body, [class*="css"] { color: var(--cv-text); }
.stApp {
  background:
    radial-gradient(ellipse at 12% 4%, rgba(37, 99, 235, .30), transparent 35%),
    radial-gradient(ellipse at 88% 12%, rgba(13, 148, 136, .20), transparent 34%),
    linear-gradient(135deg, #07111f 0%, #0b1830 52%, #10152b 100%);
  color: var(--cv-text);
}
[data-testid="stHeader"] { background: rgba(7,17,31,.45); }
.block-container { padding-top: 1.7rem; padding-bottom: 2.5rem; max-width: 1550px; }
/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(12,28,49,.97), rgba(8,19,35,.97));
  border-right: 1px solid var(--cv-line);
}
[data-testid="stSidebar"] * { color: #eaf4ff !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { color: #eaf4ff !important; font-weight: 600; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #a9bfd6 !important; }
/* Hero and glass cards */
.hero {
  padding: 1.8rem 1.9rem; border: 1px solid rgba(125,211,252,.32); border-radius: 24px;
  background: linear-gradient(120deg, rgba(37,99,235,.27), rgba(13,148,136,.16) 55%, rgba(139,92,246,.20));
  box-shadow: 0 18px 55px rgba(0,0,0,.24), inset 0 1px 0 rgba(255,255,255,.10);
  backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px); margin-bottom: 1.25rem;
}
.eyebrow { color: #8be9ff; text-transform: uppercase; letter-spacing: .18em; font-size: .75rem; font-weight: 850; }
.hero h1 { color: #ffffff !important; margin: .35rem 0 .55rem; font-size: clamp(2.1rem, 4vw, 3.15rem); letter-spacing: -.04em; text-shadow: 0 2px 22px rgba(103,232,249,.12); }
.hero p { color: #e0edff !important; font-size: 1.06rem; line-height: 1.65; margin: 0; max-width: 920px; }
.metric-card {
  background: linear-gradient(145deg, rgba(25,47,75,.82), rgba(12,27,48,.76));
  border: 1px solid rgba(148,190,226,.28); padding: 1.15rem 1.2rem; border-radius: 18px;
  min-height: 112px; box-shadow: 0 12px 30px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.08);
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
}
.metric-label { color: #b9cce2 !important; font-size: .78rem; font-weight: 750; letter-spacing: .07em; margin-bottom: .4rem; }
.metric-value { color: #ffffff !important; font-size: 1.55rem; font-weight: 850; }
.metric-sub { color: #80f0cf !important; font-size: .79rem; margin-top: .3rem; }
.section-title { color: #ffffff !important; font-size: 1.35rem; font-weight: 800; margin: 1rem 0 .35rem; }
.muted, .small-note { color: #c2d2e5 !important; }
.status-pill { display:inline-block; border:1px solid rgba(110,231,183,.55); background:rgba(16,94,75,.42); color:#a7f3d0 !important; border-radius:999px; padding:.28rem .72rem; font-size:.78rem; font-weight:800; }
/* Make all Streamlit text readable */
.stApp p, .stApp li, .stApp label, .stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stWidgetLabel"] p, .stApp .stCaption, .stApp small { color: #e6effb; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 { color: #ffffff !important; }
[data-testid="stCaptionContainer"] p { color: #b9cce2 !important; }
/* Inputs, uploaders, selectors */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
[data-baseweb="select"] > div, [data-baseweb="input"] {
  background: rgba(13,29,50,.88) !important; color: #ffffff !important;
  border: 1px solid rgba(148,190,226,.35) !important; border-radius: 11px !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: #9eb4cb !important; }
[data-testid="stFileUploader"] {
  background: rgba(18,39,65,.66); border: 1px dashed rgba(125,211,252,.58);
  border-radius: 16px; padding: .8rem; backdrop-filter: blur(12px);
}
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section *,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label *,
[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {
  color: #f4f8ff !important;
}
[data-testid="stFileUploader"] button {
  color: #ffffff !important;
  background: #1d4770 !important;
  border: 1px solid #67c9ee !important;
  border-radius: 9px !important;
  font-weight: 750 !important;
}
[data-testid="stFileUploader"] button:hover {
  background: #2563a0 !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
  background: rgba(20,45,74,.78) !important;
}
/* Buttons */
div.stButton > button, div.stDownloadButton > button {
  border-radius: 12px; border: 1px solid rgba(125,211,252,.48); font-weight: 750;
  min-height: 2.85rem; color: #f8fbff !important;
  background: rgba(30,58,92,.78); box-shadow: 0 7px 20px rgba(0,0,0,.18);
  transition: all .18s ease;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
  border-color: #67e8f9; background: rgba(37,99,235,.62); transform: translateY(-1px);
}
div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"] {
  background: linear-gradient(95deg,#2563eb,#0891b2,#0f9f83); border: 0; color: #ffffff !important;
}
/* Alerts, expanders, tabs, progress */
[data-testid="stAlert"] { background: rgba(21,43,69,.82); border: 1px solid rgba(148,190,226,.24); border-radius: 14px; }
[data-testid="stAlert"] p { color: #edf6ff !important; }
[data-testid="stExpander"] { background: rgba(17,35,58,.65); border: 1px solid var(--cv-line); border-radius: 14px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color: #f3f8ff !important; }
button[data-baseweb="tab"] { color: #c9d9eb !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #8be9ff !important; }
[role="radiogroup"] label, [data-testid="stCheckbox"] label { color: #edf5ff !important; }
hr { border-color: rgba(148,190,226,.20); }
[data-testid="stProgressBar"] > div { background: linear-gradient(90deg,#2563eb,#22c1a6); }
/* Tables and code */
[data-testid="stDataFrame"] { border: 1px solid var(--cv-line); border-radius: 12px; }
.stCode, pre, code { color: #d9efff !important; }
@media (max-width: 768px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .hero { padding: 1.25rem; }
}

/* FINAL CONTRAST OVERRIDES — paths, code panels, expanders and setup page */
.stApp [data-testid="stTextInput"] input,
.stApp [data-testid="stNumberInput"] input,
.stApp [data-testid="stTextArea"] textarea,
.stApp [data-baseweb="input"] input,
.stApp [data-baseweb="textarea"] textarea {
  background: #10243d !important;
  background-color: #10243d !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  border: 1px solid #6b91b8 !important;
  caret-color: #67e8f9 !important;
}
.stApp [data-testid="stTextInput"] > div,
.stApp [data-testid="stNumberInput"] > div,
.stApp [data-testid="stTextArea"] > div,
.stApp [data-baseweb="input"] > div,
.stApp [data-baseweb="textarea"] > div {
  background: #10243d !important;
  background-color: #10243d !important;
  border-color: #6b91b8 !important;
}
.stApp input::placeholder,
.stApp textarea::placeholder {
  color: #c2d2e5 !important;
  -webkit-text-fill-color: #c2d2e5 !important;
  opacity: 1 !important;
}
/* Streamlit code blocks (the white panels in Model & Setup) */
.stApp [data-testid="stCodeBlock"],
.stApp [data-testid="stCode"],
.stApp .stCodeBlock,
.stApp pre,
.stApp pre code,
.stApp code {
  background: #0b1b2e !important;
  background-color: #0b1b2e !important;
  color: #eaf6ff !important;
  -webkit-text-fill-color: #eaf6ff !important;
  border-color: #426486 !important;
  text-shadow: none !important;
}
.stApp [data-testid="stCodeBlock"] *,
.stApp [data-testid="stCode"] *,
.stApp pre span,
.stApp code span {
  color: #eaf6ff !important;
  -webkit-text-fill-color: #eaf6ff !important;
}
/* Expander header and Processing Details panel */
.stApp [data-testid="stExpander"],
.stApp [data-testid="stExpander"] details,
.stApp [data-testid="stExpander"] summary,
.stApp [data-testid="stExpander"] summary:hover {
  background: rgba(17, 35, 58, .96) !important;
  color: #ffffff !important;
  border-color: #547494 !important;
}
.stApp [data-testid="stExpander"] summary *,
.stApp [data-testid="stExpander"] summary p,
.stApp [data-testid="stExpander"] summary span {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}
/* Setup status text, captions, checkbox labels and all muted copy */
.stApp [data-testid="stCaptionContainer"] *,
.stApp .stCaption,
.stApp small,
.stApp [data-testid="stCheckbox"] label,
.stApp [data-testid="stCheckbox"] label *,
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li {
  color: #d7e6f7 !important;
  -webkit-text-fill-color: #d7e6f7 !important;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-testid="stSubheader"] {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}
/* Native select/dropdown values */
.stApp [data-baseweb="select"] *,
.stApp [data-baseweb="popover"] *,
.stApp [role="listbox"] *,
.stApp [role="option"] {
  color: #f4f8ff !important;
}

</style>
""", unsafe_allow_html=True)

# --------------------------- U-Net model definition ---------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ImprovedUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.upconv4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.upconv3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.upconv2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.upconv1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return torch.sigmoid(self.final_conv(d1))


# --------------------------- Helpers ---------------------------
ROOT = Path(__file__).resolve().parent

# Find the project assets whether app.py is launched from the outer or inner folder.
PROJECT_DIR_CANDIDATES = [
    ROOT,
    ROOT / "ClearVision_AI_Streamlit",
    ROOT.parent / "ClearVision_AI_Streamlit" / "ClearVision_AI_Streamlit",
]
PROJECT_DIR = next(
    (folder for folder in PROJECT_DIR_CANDIDATES if (folder / "models").is_dir()),
    ROOT,
)
MODELS_DIR = PROJECT_DIR / "models"

DEFAULT_DEHAZE = MODELS_DIR / "dehazing_nhhaze_finetuned.pth"
DEFAULT_DESMOKE = MODELS_DIR / "desmoke_model.pth"
DEFAULT_YOLO = PROJECT_DIR / "yolov8n.pt"

def device_label():
    return "CUDA GPU" if torch.cuda.is_available() else "CPU"

def resolve_path(user_path):
    if not user_path:
        return None
    p = Path(user_path).expanduser()
    return p if p.exists() and p.is_file() else None

@st.cache_resource(show_spinner="Loading restoration model…")
def load_restoration_model(checkpoint_path, device_name):
    device = torch.device(device_name)
    model = ImprovedUNet().to(device)
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Normalize checkpoint names from the training model to this app's
    # ImprovedUNet names. This changes only dictionary keys, not weights.
    mapped_state = {}
    for key, value in state.items():
        new_key = key

        # Support checkpoints saved from DataParallel.
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        # Training checkpoint used a one-item Sequential around bottleneck.
        if new_key.startswith("bottleneck.0."):
            new_key = "bottleneck." + new_key[len("bottleneck.0."):]

        # Match decoder/output layer names used by the dashboard model.
        rename_prefixes = {
            "up4.": "upconv4.",
            "up3.": "upconv3.",
            "up2.": "upconv2.",
            "up1.": "upconv1.",
            "output.": "final_conv.",
        }
        for old_prefix, new_prefix in rename_prefixes.items():
            if new_key.startswith(old_prefix):
                new_key = new_prefix + new_key[len(old_prefix):]
                break

        # Older PyTorch checkpoints may contain this bookkeeping buffer,
        # which is not needed for inference.
        if new_key.endswith(".num_batches_tracked"):
            continue

        mapped_state[new_key] = value

    # Keep strict loading: if the actual layer shapes/architecture differ,
    # report that rather than silently producing an invalid restoration.
    model.load_state_dict(mapped_state, strict=True)
    model.eval()
    return model

def image_to_tensor(rgb):
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().unsqueeze(0)

def tiled_restore(image_rgb, model, device, tile_size=256, overlap=64):
    """Run restoration on original-resolution tiles and blend overlapping areas."""
    h, w, _ = image_rgb.shape
    stride = tile_size - overlap
    pad_h, pad_w = max(0, tile_size - h), max(0, tile_size - w)
    if pad_h or pad_w:
        mode = "reflect" if min(h, w) > 1 else "edge"
        image_rgb = np.pad(image_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
    H, W, _ = image_rgb.shape
    out = np.zeros((H, W, 3), dtype=np.float32)
    weights = np.zeros((H, W, 1), dtype=np.float32)
    win1d = np.hanning(tile_size + 2)[1:-1].astype(np.float32)
    window = np.maximum(np.outer(win1d, win1d)[..., None], 1e-3)
    ys = list(range(0, max(H - tile_size, 0) + 1, stride))
    xs = list(range(0, max(W - tile_size, 0) + 1, stride))
    if not ys or ys[-1] != H - tile_size:
        ys.append(H - tile_size)
    if not xs or xs[-1] != W - tile_size:
        xs.append(W - tile_size)
    with torch.inference_mode():
        for y in ys:
            for x in xs:
                patch = image_rgb[y:y+tile_size, x:x+tile_size]
                tensor = image_to_tensor(patch).to(device)
                pred = model(tensor).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
                out[y:y+tile_size, x:x+tile_size] += pred * window
                weights[y:y+tile_size, x:x+tile_size] += window
    return np.clip(out / np.maximum(weights, 1e-8), 0, 1)[:h, :w, :]

def save_pil_png(rgb_array, path):
    Image.fromarray(np.clip(rgb_array * 255, 0, 255).astype(np.uint8)).save(path)

def metric_card(label, value, sub=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )

def hero(title, subtitle, tag="COMPUTER VISION SUITE"):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{tag}</div><h1>{title}</h1>'
        f'<p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.markdown("## 🔭 ClearVision AI")
    st.caption("RESTORE • DETECT • ANALYZE")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Image Dehazing", "Image Desmoking", "Video Person Detection", "Evaluation Lab", "Model & Setup"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("**Runtime**")
    st.markdown(f'<span class="status-pill">{device_label()}</span>', unsafe_allow_html=True)
    st.caption("Local-first inference • Uploads processed in this app session")
    with st.expander("Model paths", expanded=False):
        dehaze_path_text = st.text_input("Dehazing checkpoint", str(DEFAULT_DEHAZE))
        desmoke_path_text = st.text_input("Desmoking checkpoint", str(DEFAULT_DESMOKE))
        yolo_path_text = st.text_input("YOLO weights", str(DEFAULT_YOLO))
    st.markdown("---")
    st.caption("Prototype for research demonstration. Verify outputs before operational use.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------- Overview ---------------------------
if page == "Overview":
    hero("See through the haze.", "An integrated AI vision workspace for image restoration and people detection in degraded visual conditions.")
    a, b, c, d = st.columns(4)
    with a: metric_card("VISION MODULES", "03", "Dehaze • Desmoke • Detect")
    with b: metric_card("RESTORATION ENGINE", "U-Net", "Paired-image learning")
    with c: metric_card("PERSON DETECTOR", "YOLO", "Frame-by-frame detection")
    with d: metric_card("INFERENCE DEVICE", device_label(), "Detected automatically")
    st.markdown('<div class="section-title">Choose a workspace</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown("### 🌫️ Image Dehazing")
        st.write("Restore visibility in foggy or hazy images using the NH-HAZE fine-tuned checkpoint.")
        if st.button("Open dehazing →", use_container_width=True, key="go_dehaze"):
            st.session_state["nav_target"] = "Image Dehazing"
            st.rerun()
    with c2:
        st.markdown("### 💨 Image Desmoking")
        st.write("Reduce smoke degradation with your SmokeBench-trained restoration checkpoint.")
        if st.button("Open desmoking →", use_container_width=True, key="go_desmoke"):
            st.session_state["nav_target"] = "Image Desmoking"
            st.rerun()
    with c3:
        st.markdown("### 🎥 Video Person Detection")
        st.write("Detect and optionally track people in uploaded video with red bounding boxes.")
        if st.button("Open video detection →", use_container_width=True, key="go_video"):
            st.session_state["nav_target"] = "Video Person Detection"
            st.rerun()
    st.info("For a panel demo: show one hazy image, one smoky image, then a short video. Explain the model, metrics, limitations, and intended use.")

# Make overview shortcut buttons functional on rerun.
if "nav_target" in st.session_state:
    target = st.session_state.pop("nav_target")
    # Sidebar radio state can be updated only before widget creation, so use query-param-like message.
    st.info(f"Use the sidebar to select **{target}**.")

# --------------------------- Image restoration pages ---------------------------
elif page in ["Image Dehazing", "Image Desmoking"]:
    is_dehaze = page == "Image Dehazing"
    title = "Image Dehazing" if is_dehaze else "Image Desmoking"
    subtitle = ("Reduce atmospheric haze and improve scene visibility."
                if is_dehaze else "Reduce smoke-related visibility loss in an image.")
    hero(title, subtitle, "IMAGE RESTORATION")

    checkpoint_text = dehaze_path_text if is_dehaze else desmoke_path_text
    checkpoint = resolve_path(checkpoint_text)

    st.markdown(
        '<div class="section-title">Upload your input image</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="muted">Choose a hazy image for dehazing or a smoky image for desmoking. '
        'Supported formats: PNG, JPG, JPEG and WEBP.</p>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Browse your computer and select an image",
        type=["png", "jpg", "jpeg", "webp"],
        key=f"upload_{page}",
        help="Select an image from your computer. The restored result will appear below.",
    )

    if checkpoint is None:
        st.warning("The restoration checkpoint was not found. The image upload is available, but restoration cannot run until the checkpoint path is corrected.")
        st.caption("Expected checkpoint:")
        st.code(str(DEFAULT_DEHAZE if is_dehaze else DEFAULT_DESMOKE), language="text")

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("**Input image**")
        st.caption("Your selected original image")
    with right:
        st.markdown("**Restored output**")
        st.caption("AI-restored image will appear here")

    if uploaded and checkpoint is not None:
        try:
            pil_image = Image.open(uploaded).convert("RGB")
            rgb = np.asarray(pil_image).astype(np.float32) / 255.0
            with st.spinner("Loading model and restoring image in overlapping tiles…"):
                model = load_restoration_model(str(checkpoint.resolve()), DEVICE)
                restored = tiled_restore(rgb, model, DEVICE)
            with left:
                st.image(pil_image, use_container_width=True)
            with right:
                st.image(restored, clamp=True, use_container_width=True)
            st.success(f"Processed {pil_image.width} × {pil_image.height} image.")
            out_bytes = Image.fromarray((restored * 255).round().astype(np.uint8))
            import io
            buffer = io.BytesIO()
            out_bytes.save(buffer, format="PNG")
            st.download_button(
                "⬇ Download restored image",
                data=buffer.getvalue(),
                file_name=("dehazed_" if is_dehaze else "desmoked_") + Path(uploaded.name).stem + ".png",
                mime="image/png",
                type="primary",
                use_container_width=True,
            )
            with st.expander("Processing details"):
                st.write(f"Checkpoint: `{checkpoint.name}`")
                st.write(f"Image size: `{pil_image.width} × {pil_image.height}`")
                st.write("Inference: 256 × 256 tiles with 64-pixel overlap and weighted blending.")
                st.caption("Restoration quality varies with the image domain. Inspect results for artifacts.")
        except Exception as exc:
            st.error(f"Could not process this image: {exc}")
            st.caption("Check that the checkpoint matches the included ImprovedUNet architecture.")
    elif uploaded and checkpoint is None:
        st.info("Your image is selected. Correct the checkpoint path in the sidebar under Model paths to run restoration.")

# --------------------------- Video person detection ---------------------------
elif page == "Video Person Detection":
    hero("Video Person Detection", "Upload a clip to detect people and optionally track them across frames.", "VIDEO ANALYTICS")
    if YOLO is None:
        st.error("Ultralytics is not installed. Install requirements with: pip install -r requirements.txt")
        st.stop()
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
    c1, c2, c3 = st.columns(3)
    with c1: conf = st.slider("Detection confidence", 0.10, 0.90, 0.25, 0.05)
    with c2: max_width = st.select_slider("Processing width", options=[480, 640, 800, 960], value=640)
    with c3: tracking = st.toggle("Enable tracking", value=True)
    st.caption("For a smooth demo, start with a short clip (10–30 seconds). Processing time depends on video length and hardware.")
    if uploaded_video and st.button("▶ Run person detection", type="primary", use_container_width=True):
        temp_dir = Path(tempfile.mkdtemp(prefix="clearvision_"))
        input_path = temp_dir / Path(uploaded_video.name).name
        output_path = temp_dir / "clearvision_person_detection.mp4"
        input_path.write_bytes(uploaded_video.getbuffer())
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 120:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            cap.release()
            st.error("Could not read this video. Try an MP4 or AVI file.")
            st.stop()
        out_w = min(width, max_width)
        out_h = int(height * out_w / width)
        out_h = max(2, out_h - out_h % 2)
        out_w = max(2, out_w - out_w % 2)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            st.error("Could not initialize video writer on this system.")
            st.stop()
        try:
            yolo_model = YOLO(str(resolve_path(yolo_path_text) or yolo_path_text))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            progress = st.progress(0, text="Preparing video…")
            status = st.empty()
            frame_count, people_total, max_people = 0, 0, 0
            preview = st.empty()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                if tracking:
                    result = yolo_model.track(frame, persist=True, classes=[0], conf=conf, tracker="bytetrack.yaml", verbose=False)[0]
                else:
                    result = yolo_model.predict(frame, classes=[0], conf=conf, verbose=False)[0]
                count = 0
                if result.boxes is not None:
                    for box_index, box in enumerate(result.boxes):
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = xyxy.tolist()
                        score = float(box.conf[0].cpu())
                        track_id = None
                        if tracking and getattr(result.boxes, "id", None) is not None:
                            ids = result.boxes.id
                            if ids is not None:
                                track_id = int(ids[box_index].item())
                        label = f"Person {score:.2f}" if track_id is None else f"Person #{track_id} {score:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 40, 240), 2)
                        cv2.rectangle(frame, (x1, max(0, y1-26)), (min(out_w-1, x1+max(120, len(label)*8)), y1), (30, 40, 240), -1)
                        cv2.putText(frame, label, (x1+4, max(16, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        count += 1
                people_total += count
                max_people = max(max_people, count)
                writer.write(frame)
                frame_count += 1
                if frame_count % 10 == 0:
                    preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Live processing preview • frame {frame_count}", use_container_width=True)
                if total_frames:
                    progress.progress(min(frame_count / total_frames, 1.0), text=f"Processing frame {frame_count}/{total_frames}")
                status.caption(f"Frames processed: {frame_count} · People detected in current frame: {count}")
            progress.progress(1.0, text="Video processing complete")
            cap.release()
            writer.release()
            if frame_count == 0:
                st.error("No frames could be processed.")
            else:
                st.success("Person detection completed.")
                m1, m2, m3 = st.columns(3)
                with m1: metric_card("FRAMES PROCESSED", f"{frame_count:,}")
                with m2: metric_card("DETECTIONS ACROSS FRAMES", f"{people_total:,}")
                with m3: metric_card("MAX PEOPLE IN A FRAME", str(max_people))
                st.video(str(output_path))
                st.download_button("⬇ Download annotated video", data=output_path.read_bytes(), file_name="clearvision_person_detection.mp4", mime="video/mp4", type="primary", use_container_width=True)
                st.caption("Detection counts are per frame, not unique people. Tracking IDs may reset between videos.")
        except Exception as exc:
            st.error(f"Video processing failed: {exc}")
            st.caption("Check the YOLO weights path, video codec, and installed dependencies.")
        finally:
            cap.release()
            writer.release()

# --------------------------- Evaluation Lab ---------------------------
elif page == "Evaluation Lab":
    hero("Evaluation Lab", "Measure restoration against a paired clean reference instead of relying only on visual impressions.", "QUALITY & VALIDATION")
    st.markdown("### Compare restored image against clean ground truth")
    st.write("Upload the model output and its matching clean target. Both images should depict the same scene and have the same dimensions.")
    c1, c2 = st.columns(2)
    with c1: pred_file = st.file_uploader("Restored image", type=["png", "jpg", "jpeg"], key="eval_pred")
    with c2: gt_file = st.file_uploader("Clean ground truth", type=["png", "jpg", "jpeg"], key="eval_gt")
    if pred_file and gt_file:
        try:
            from skimage.metrics import peak_signal_noise_ratio, structural_similarity
            pred = np.asarray(Image.open(pred_file).convert("RGB").resize((256,256))).astype(np.float32)/255.0
            gt = np.asarray(Image.open(gt_file).convert("RGB").resize((256,256))).astype(np.float32)/255.0
            psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
            ssim = structural_similarity(gt, pred, data_range=1.0, channel_axis=2)
            m1, m2 = st.columns(2)
            with m1: metric_card("PSNR", f"{psnr:.3f} dB", "Higher is generally better")
            with m2: metric_card("SSIM", f"{ssim:.4f}", "Closer to 1 is generally better")
            st.image([pred, gt], caption=["Restored image (resized for evaluation)", "Clean reference (resized for evaluation)"], use_container_width=True)
            st.caption("Images are resized to 256 × 256 for metric calculation in this prototype. Use the same evaluation procedure when comparing models.")
        except Exception as exc:
            st.error(f"Metric calculation failed: {exc}")
    st.info("PSNR and SSIM require a matching clean reference. Do not report these metrics for arbitrary images without ground truth.")

# --------------------------- Model & setup ---------------------------
elif page == "Model & Setup":
    hero("Model & Setup", "Check required checkpoints and prepare a reliable local demo environment.", "SYSTEM STATUS")
    rows = [
        ("NH-HAZE fine-tuned dehazing", dehaze_path_text),
        ("SmokeBench desmoking", desmoke_path_text),
        ("YOLO person detector weights", yolo_path_text),
    ]
    for label, path_text in rows:
        p = resolve_path(path_text)
        c1, c2 = st.columns([3, 1])
        with c1:
            st.write(f"**{label}**")
            st.code(path_text, language=None)
        with c2:
            st.write("✅ Found" if p else "⚠️ Not found")
    st.markdown("### Recommended project folder")
    st.code("""ClearVision_AI_Streamlit/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── dehazing_nhhaze_finetuned.pth
│   └── desmoke_model.pth
└── yolov8n.pt  # downloaded automatically by Ultralytics if absent""", language="text")
    st.markdown("### Demo checklist")
    st.checkbox("Test image dehazing with a hazy image", key="check_dehaze")
    st.checkbox("Test image desmoking with a smoky image", key="check_desmoke")
    st.checkbox("Test person detection with a short video", key="check_video")
    st.checkbox("Check downloads and explain model limitations", key="check_downloads")
    st.warning("This is a research prototype. Do not use as the sole basis for safety-critical surveillance decisions.")
