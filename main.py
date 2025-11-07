# main.py — AROGYAM (Final Stable Build)
# Tabs: Diagnosis | Research Context | Downloads
# Research Context tab shows Gemini’s section 7 or verified URLs for the diagnosed condition.

import os
import io
import re
from pathlib import Path
import streamlit as st
from PIL import Image as PILImage, ImageOps as PILImageOps
import numpy as np
from dotenv import load_dotenv
from fpdf import FPDF
from gtts import gTTS

# Optional DICOM
try:
    import pydicom
except Exception:
    pydicom = None

import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util

# Project imports
from prompt_config import PROMPT
from report_prompt import REPORT_PROMPT
from knowledge_base import search_kb

# -----------------------------
# UI Configuration
# -----------------------------
st.set_page_config(page_title="AROGYAM — Agentic AI for Medical Imaging and Diagnosis", layout="wide")

# Dark Theme Styling
st.markdown("""
<style>
body { background-color: #0E1117; color: #FAFAFA; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827 !important;
    color: #FAFAFA !important;
    border-right: 1px solid #222;
}
.sidebar-title {
    font-weight: 700;
    font-size: 18px;
    color: #E5E7EB;
    margin-bottom: 0.75rem;
}
label, span, p {
    color: #E5E7EB !important;
}
textarea, input, div[data-baseweb="input"] > div {
    background-color: #1F2937 !important;
    color: #FAFAFA !important;
    border: 1px solid #374151 !important;
    border-radius: 6px !important;
}

/* Buttons */
div.stButton > button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    height: 2.6rem !important;
    width: 100%;
}
div.stButton > button:hover {
    background-color: #1E40AF !important;
}

/* Report container */
.report-box {
    background-color: #1F2937;
    border-radius: 12px;
    padding: 20px;
    color: #E5E7EB;
    border: 1px solid #374151;
    box-shadow: 0 2px 4px rgba(0,0,0,0.4);
}

/* Links */
.link-item a {
    color: #60A5FA !important;
    text-decoration: none;
}
.link-item a:hover {
    text-decoration: underline;
}

/* Tabs and headers */
h1, h2, h3 {
    color: #E5E7EB;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Constants
# -----------------------------
APP_TITLE = "🩺 AROGYAM — Agentic AI for Medical Imaging and Diagnosis"
DISCLAIMER = (
    "⚠️ This analysis is for educational and informational purposes only. "
    "It is not a medical diagnosis or prescription. "
    "Always consult a qualified healthcare professional for any medical concerns."
)
MODEL_NAME = "models/gemini-2.0-flash"

# -----------------------------
# Utility Functions
# -----------------------------
def _scrub_medication_doses(text: str) -> str:
    if not text:
        return text
    pattern = re.compile(r"(\b\d+(\.\d+)?\s*(mg|mcg|g|gram|ml|mL|units|IU)\b[^.\n]*)", re.I)
    return pattern.sub("[dose removed — clinician required]", text)

def _extract_research_context(text: str) -> str:
    """Extract section ### 7. Research Context from Gemini output."""
    match = re.search(r"###\s*7\.\s*Research Context[\s\S]*?(?=\n###|\Z)", text, re.I)
    return match.group(0).strip() if match else ""

def _extract_condition_name(text: str) -> str:
    """Extract condition name from Diagnosis Summary line."""
    match = re.search(r"(?i)diagnosis summary[:\-]?\s*([A-Za-z\s]+)", text)
    return match.group(1).strip() if match else ""

def _generate_condition_links(condition: str) -> list[str]:
    """Generate verified reference URLs for the diagnosed condition."""
    cond_q = condition.replace(" ", "+")
    return [
        f"• [WHO on {condition}](https://www.who.int/search?q={cond_q})",
        f"• [NIH: {condition}](https://www.nih.gov/search?utf8=✓&affiliate=nih&query={cond_q})",
        f"• [CDC Information on {condition}](https://www.cdc.gov/search/?query={cond_q})",
        f"• [PubMed Articles on {condition}](https://pubmed.ncbi.nlm.nih.gov/?term={cond_q})",
        f"• [Mayo Clinic – {condition}](https://www.mayoclinic.org/search/search-results?q={cond_q})",
        f"• [Cleveland Clinic: {condition}](https://my.clevelandclinic.org/search?q={cond_q})",
        f"• [MedlinePlus – {condition}](https://medlineplus.gov/search/?query={cond_q})",
        f"• [WebMD – {condition}](https://www.webmd.com/search/search_results/default.aspx?query={cond_q})",
        f"• [Healthline – {condition}](https://www.healthline.com/search?q1={cond_q})",
        f"• [Johns Hopkins Medicine – {condition}](https://www.hopkinsmedicine.org/search/?q={cond_q})"
    ]

def _load_image_any(file: io.BytesIO, filename: str) -> PILImage.Image:
    suffix = Path(filename).suffix.lower()
    if suffix == ".dcm":
        if pydicom is None:
            raise RuntimeError("pydicom not installed.")
        ds = pydicom.dcmread(file)
        arr = ds.pixel_array.astype(np.float32)
        arr = 255 * (arr - arr.min()) / (arr.ptp() or 1)
        arr8 = arr.astype(np.uint8)
        return PILImage.fromarray(arr8)
    img = PILImage.open(file)
    img = PILImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img

def _resize_for_display(img: PILImage.Image, max_side: int = 1024) -> PILImage.Image:
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), resample=PILImage.LANCZOS)
    return img

def _image_bytes(img: PILImage.Image, fmt="JPEG", quality=90) -> bytes:
    bio = io.BytesIO()
    img.save(bio, format=fmt, quality=quality)
    return bio.getvalue()

@st.cache_resource
def _get_clip_model():
    return SentenceTransformer("clip-ViT-B-32")

def _check_prompt_image_similarity(prompt_text: str, image: PILImage.Image) -> bool:
    model = _get_clip_model()
    text_emb = model.encode([prompt_text], convert_to_tensor=True)
    image_emb = model.encode([image], convert_to_tensor=True)
    score = util.cos_sim(text_emb, image_emb).item()
    return score >= 0.25

@st.cache_data(show_spinner=False)
def _run_llm(system_prompt: str, context_text: str, inputs: list):
    """Run Google Gemini model."""
    model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_prompt)
    result = model.generate_content(
        contents=inputs,
        safety_settings=None,
        generation_config={"temperature": 0.25, "top_p": 0.9},
    )
    return result.text or ""

# -----------------------------
# App Setup
# -----------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY", "")
if not api_key:
    st.error("GOOGLE_API_KEY not set.")
else:
    genai.configure(api_key=api_key)

st.markdown(f"## {APP_TITLE}")
st.caption(DISCLAIMER)

# Sidebar
st.sidebar.markdown("<p class='sidebar-title'>⚙️ Select Mode</p>", unsafe_allow_html=True)
mode = st.sidebar.radio("", ["Medical Image Analysis", "Health Report Analysis"])
st.sidebar.markdown("<p class='sidebar-title'>🧠 Your Prompt</p>", unsafe_allow_html=True)
user_prompt = st.sidebar.text_area("Describe your condition or question")

if mode == "Medical Image Analysis":
    uploaded = st.sidebar.file_uploader("📷 Upload Medical Image", type=["jpg","jpeg","png","bmp","webp","dcm"])
    run_btn = st.sidebar.button("🔍 Analyze Image")
else:
    uploaded = st.sidebar.file_uploader("📄 Upload Health Report", type=["pdf","txt"])
    run_btn = st.sidebar.button("🔍 Analyze Report")

# Tabs
tab_diag, tab_research, tab_downloads = st.tabs(["🩻 Diagnosis", "📚 Research Context", "⬇️ Downloads"])

# Session vars
if "safe_text" not in st.session_state:
    st.session_state.safe_text = ""
if "img_disp" not in st.session_state:
    st.session_state.img_disp = None
if "research_context_text" not in st.session_state:
    st.session_state.research_context_text = ""

# -----------------------------
# Analysis Logic
# -----------------------------
if run_btn:
    if not api_key:
        st.error("Missing GOOGLE_API_KEY.")
        st.stop()
    if not user_prompt.strip():
        st.error("Enter a valid prompt.")
        st.stop()

    kb_refs = search_kb(user_prompt, top_k=3) or []
    kb_context = "\n\n".join(kb_refs)

    context_text = f"User Prompt: {user_prompt}\n\n[INTERNAL_KB]\n{kb_context}\n[/INTERNAL_KB]"

    if mode == "Medical Image Analysis":
        if not uploaded:
            st.error("Upload an image first.")
            st.stop()
        img = _load_image_any(io.BytesIO(uploaded.getvalue()), uploaded.name)
        img_disp = _resize_for_display(img)
        img_bytes = _image_bytes(img_disp)
        inputs = [{"text": context_text}, {"inline_data": {"mime_type": "image/jpeg", "data": img_bytes}}]
        with st.spinner("Analyzing image..."):
            result_text = _run_llm(PROMPT, context_text, inputs)
        st.session_state.img_disp = img_disp
    else:
        if not uploaded:
            st.error("Upload a report first.")
            st.stop()
        suffix = Path(uploaded.name).suffix.lower()
        if suffix == ".pdf":
            inputs = [{"text": context_text}, {"inline_data": {"mime_type": "application/pdf", "data": uploaded.getvalue()}}]
        else:
            report_text = uploaded.read().decode("utf-8", errors="ignore")
            context_text += "\n\nReport Text:\n" + report_text
            inputs = [{"text": context_text}]
        with st.spinner("Analyzing report..."):
            result_text = _run_llm(REPORT_PROMPT, context_text, inputs)

    st.session_state.safe_text = _scrub_medication_doses(result_text)
    st.session_state.research_context_text = _extract_research_context(result_text)

# -----------------------------
# Tabs Display
# -----------------------------
with tab_diag:
    if mode == "Medical Image Analysis" and st.session_state.img_disp:
        st.image(st.session_state.img_disp, caption="Uploaded Image", use_container_width=True)
    if st.session_state.safe_text:
        st.markdown("### 🧾 Diagnostic Report")
        st.markdown(f"<div class='report-box'>{st.session_state.safe_text}</div>", unsafe_allow_html=True)
        st.info(DISCLAIMER)

with tab_research:
    st.markdown("### 🌐 Research Context — Verified Medical References")

    if st.session_state.get("research_context_text"):
        st.markdown(st.session_state.research_context_text)
    else:
        diagnosed_condition = _extract_condition_name(st.session_state.safe_text)
        if diagnosed_condition:
            st.markdown(f"#### Verified Web References for **{diagnosed_condition}**")
            links = _generate_condition_links(diagnosed_condition)
            for ref in links:
                st.markdown(ref)
            st.info(f"These sources provide authoritative, peer-reviewed information about {diagnosed_condition}.")
        else:
            st.markdown("#### Default Trusted Healthcare References")
            default_refs = [
                "• [World Health Organization (WHO)](https://www.who.int/)",
                "• [National Institutes of Health (NIH)](https://www.nih.gov/)",
                "• [Centers for Disease Control and Prevention (CDC)](https://www.cdc.gov/)",
                "• [PubMed Biomedical Database](https://pubmed.ncbi.nlm.nih.gov/)",
                "• [Mayo Clinic](https://www.mayoclinic.org/)",
                "• [Cleveland Clinic](https://my.clevelandclinic.org/)",
                "• [MedlinePlus](https://medlineplus.gov/)",
                "• [WebMD](https://www.webmd.com/)",
                "• [Healthline](https://www.healthline.com/)",
                "• [Johns Hopkins Medicine](https://www.hopkinsmedicine.org/)"
            ]
            for ref in default_refs:
                st.markdown(ref)
            st.info("These verified organizations provide authoritative health information globally.")

with tab_downloads:
    if st.session_state.safe_text:
        st.markdown("### 📂 Generate Outputs")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Generate PDF Report"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                for line in st.session_state.safe_text.splitlines():
                    pdf.multi_cell(0, 8, line if line.strip() else "")
                pdf_path = "Arogyam_Report.pdf"
                pdf.output(pdf_path)
                with open(pdf_path, "rb") as f:
                    st.download_button("⬇️ Download PDF", f, file_name="Arogyam_Report.pdf")
        with col2:
            if st.button("🔊 Generate Audio Summary"):
                tts = gTTS(text=st.session_state.safe_text[:4000], lang="en")
                audio_path = "Arogyam_Audio.mp3"
                tts.save(audio_path)
                with open(audio_path, "rb") as f:
                    st.download_button("⬇️ Download Audio", f, file_name="Arogyam_Audio.mp3")
    else:
        st.info("Run an analysis to enable downloads.")
