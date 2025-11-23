# main.py — AROGYAM (Final Clean Version with Structured CSS + Patient Details Stage)

import os, io, re, requests
from urllib.parse import urlparse
from pathlib import Path
import streamlit as st
from PIL import Image as PILImage, ImageOps as PILImageOps
import numpy as np
from dotenv import load_dotenv
from fpdf import FPDF
from gtts import gTTS
from duckduckgo_search import DDGS
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util

from prompt_config import PROMPT
from report_prompt import REPORT_PROMPT
from knowledge_base import search_kb

try:
    import pydicom
except:
    pydicom = None


# =========================================================
# Streamlit Page Configuration
# =========================================================
st.set_page_config(
    page_title="AROGYAM — Agentic AI for Medical Imaging and Diagnosis",
    layout="wide"
)

# =========================================================
# Structured CSS (Clean & Organized)
# =========================================================
st.markdown("""
<style>

/* ===========================
   GLOBAL PAGE STYLING
   =========================== */
body {
    background-color: #0E1117;
    color: #FAFAFA;
    font-family: "Inter", sans-serif;
}

/* ===========================
   SIDEBAR STYLING
   =========================== */
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

/* ===========================
   FORM INPUTS
   =========================== */
textarea,
input,
div[data-baseweb="input"] > div {
    background-color: #1F2937 !important;
    color: #FAFAFA !important;
    border: 1px solid #374151 !important;
    border-radius: 6px !important;
}

label, span, p {
    color: #E5E7EB !important;
}

/* ===========================
   BUTTONS
   =========================== */
div.stButton > button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    height: 2.6rem !important;
    width: 100%;
    transition: background-color 0.15s ease-in-out;
}

div.stButton > button:hover {
    background-color: #1E40AF !important;
}

/* ===========================
   REPORT BOX
   =========================== */
.report-box {
    background-color: #1F2937;
    border-radius: 12px;
    padding: 20px;
    color: #E5E7EB;
    border: 1px solid #374151;
    box-shadow: 0 2px 4px rgba(0,0,0,0.4);
    position: relative;
    z-index: 50;
}

/* ===========================
   HEADINGS
   =========================== */
h1, h2, h3, h4 {
    color: #E5E7EB !important;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# Constants
# =========================================================
APP_TITLE = "🩺 AROGYAM — Agentic AI for Medical Imaging and Diagnosis"
DISCLAIMER = (
    "⚠️ This report is for educational use only. Not a medical prescription. "
    "Consult certified healthcare professionals."
)

MODEL_NAME = "models/gemini-2.0-flash"
CLIP_THRESHOLD = 0.25
MAX_IMAGE_MB = 10


# =========================================================
# Utility Functions
# =========================================================
def _scrub_medication_doses(text: str) -> str:
    if not text:
        return text
    pat = re.compile(r"(\b\d+(\.\d+)?\s*(mg|mcg|g|gram|ml|mL|units|IU)\b[^.\n]*)", re.I)
    return pat.sub("[dose removed — clinician required]", text)


def _extract_condition_name(text: str) -> str:
    if not text: 
        return ""
    m = re.search(r"(?i)diagnosis summary[:\-]?\s*([A-Za-z][A-Za-z \-/]+)", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"\b([A-Z][a-z]+(?:itis|osis|emia|opathy|oma|algia|dermatitis|psoriasis|asthma|pneumonia))\b", text)
    return m.group(1).strip() if m else ""


@st.cache_data(ttl=86400)
def _fetch_verified_references(disease_name: str, max_results: int = 10):
    query = (
        f'"{disease_name}" (research OR study OR clinical trial) '
        "site:who.int OR site:nih.gov OR site:pubmed.ncbi.nlm.nih.gov OR site:cdc.gov "
        "OR site:mayoclinic.org OR site:clevelandclinic.org OR site:nature.com "
        "OR site:springer.com OR site:thelancet.com OR site:nejm.org"
    )

    verified = [
        "who.int","nih.gov","pubmed.ncbi.nlm.nih.gov","cdc.gov",
        "mayoclinic.org","clevelandclinic.org","nature.com",
        "springer.com","thelancet.com","nejm.org"
    ]

    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results*3):
            title = (r.get("title") or "").strip()
            url = (r.get("href") or "").strip()
            snippet = (r.get("body") or "").strip()

            if not title or not url:
                continue

            domain = urlparse(url).netloc.replace("www.", "")
            if not any(domain.endswith(v) for v in verified):
                continue

            try:
                resp = requests.get(url, timeout=4, stream=True)
                if not (200 <= resp.status_code < 300):
                    continue
            except:
                continue

            results.append(f"**[{title}]({url})**\n_{snippet}_")

            if len(results) >= max_results:
                break

    return results


def _load_image_any(file: io.BytesIO, filename: str) -> PILImage.Image:
    ext = Path(filename).suffix.lower()

    if ext == ".dcm":
        if pydicom is None:
            raise RuntimeError("pydicom not installed.")
        ds = pydicom.dcmread(file)
        arr = ds.pixel_array.astype(np.float32)
        arr = 255 * (arr - arr.min()) / (arr.ptp() or 1)
        return PILImage.fromarray(arr.astype(np.uint8))

    img = PILImage.open(file)
    img = PILImageOps.exif_transpose(img)
    return img.convert("RGB")


def _resize_for_display(img: PILImage.Image, max_side: int = 1024):
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1:
        return img.resize((int(w*scale), int(h*scale)))
    return img


def _image_bytes(img: PILImage.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=90)
    return b.getvalue()


@st.cache_resource
def _get_clip():
    return SentenceTransformer("clip-ViT-B-32")


def _clip_similarity(prompt_text, image):
    model = _get_clip()
    te = model.encode([prompt_text], convert_to_tensor=True)
    ie = model.encode([image], convert_to_tensor=True)
    return float(util.cos_sim(te, ie).item())


@st.cache_data(show_spinner=False)
def _run_llm(system_prompt, context_text, inputs):
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=system_prompt
    )
    response = model.generate_content(
        contents=inputs,
        generation_config={"temperature": 0.25, "top_p": 0.9}
    )
    return response.text or ""


# =========================================================
# Environment Initialization
# =========================================================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY", "")
if not api_key:
    st.error("GOOGLE_API_KEY not set.")
else:
    genai.configure(api_key=api_key)


# =========================================================
# Page Header
# =========================================================
st.markdown(f"## {APP_TITLE}")
st.caption(DISCLAIMER)


# =========================================================
# Sidebar Input Section
# =========================================================
st.sidebar.markdown("<p class='sidebar-title'>⚙️ Mode</p>", unsafe_allow_html=True)
mode = st.sidebar.radio("", ["Medical Image Analysis", "Health Report Analysis"])

st.sidebar.markdown("<p class='sidebar-title'>⚠️ Important</p>", unsafe_allow_html=True)
st.sidebar.info("If the uploaded image and your prompt are unrelated, the diagnosis may be inaccurate.")

user_prompt = st.sidebar.text_area("Describe your concern", height=90)

extra_symptoms = st.sidebar.text_area(
    "Optional symptoms (ignored for CLIP check)",
    height=80,
    placeholder="e.g., mild fever, itching..."
)

uploaded = st.sidebar.file_uploader(
    "Upload File",
    type=["jpg","jpeg","png","bmp","webp","dcm","pdf","txt"]
)

run_btn = st.sidebar.button("Analyze")


# =========================================================
# Session State Management
# =========================================================
if "clip_passed" not in st.session_state:
    st.session_state.clip_passed = False

if "patient_details_ok" not in st.session_state:
    st.session_state.patient_details_ok = False

if "safe_text" not in st.session_state:
    st.session_state.safe_text = ""

if "img_disp" not in st.session_state:
    st.session_state.img_disp = None


# =========================================================
# STAGE 1 — CLIP SIMILARITY VALIDATION
# =========================================================
if run_btn:
    if not uploaded:
        st.error("Upload a file.")
        st.stop()

    if mode != "Medical Image Analysis":
        st.error("This feature works only for medical images.")
        st.stop()

    if not user_prompt.strip():
        st.error("Enter a valid prompt.")
        st.stop()

    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in [".jpg",".jpeg",".png",".bmp",".webp",".dcm"]:
        st.error("Upload a valid medical image.")
        st.stop()

    img = _load_image_any(io.BytesIO(uploaded.getvalue()), uploaded.name)
    img_disp = _resize_for_display(img)
    st.session_state.img_disp = img_disp

    with st.spinner("Checking prompt–image similarity..."):
        clip_score = _clip_similarity(user_prompt, img_disp)

    st.caption(f"CLIP Similarity: {clip_score:.3f} (Threshold ≥ {CLIP_THRESHOLD})")

    if clip_score < CLIP_THRESHOLD:
        st.warning("Image and prompt appear unrelated. Diagnosis stopped.")
        st.stop()
    else:
        st.success("CLIP validation passed. Now fill patient details.")
        st.session_state.clip_passed = True


# =========================================================
# STAGE 2 — PATIENT BASIC DETAILS (Required)
# =========================================================
if st.session_state.clip_passed and not st.session_state.patient_details_ok:
    st.markdown("## 🧍 Patient Basic Details")

    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])
    weight = st.number_input("Weight (kg)", min_value=1, max_value=300, step=1)
    past_diseases = st.text_area("Past diseases (optional)", placeholder="e.g., diabetes, asthma")

    def validate():
        if gender == "Select":
            st.error("Select a valid gender.")
            return False
        return True

    if st.button("Proceed to Diagnosis"):
        if not validate():
            st.stop()

        st.session_state.patient_details_ok = True
        st.session_state.age = age
        st.session_state.gender = gender
        st.session_state.weight = weight
        st.session_state.past_diseases = past_diseases
        st.success("Details saved. Running diagnosis...")


# =========================================================
# STAGE 3 — FINAL DIAGNOSIS (Image + Prompt + Patient Details)
# =========================================================
if st.session_state.clip_passed and st.session_state.patient_details_ok:

    kb_refs = search_kb(user_prompt, top_k=3) or []
    kb_text = "\n".join(kb_refs)

    symptoms_block = f"\n\nAdditional Symptoms:\n{extra_symptoms}" if extra_symptoms.strip() else ""

    context_text = (
        f"User Prompt: {user_prompt}{symptoms_block}\n\n"
        f"[USER_HEALTH_DETAILS]\n"
        f"Age: {st.session_state.age}\n"
        f"Gender: {st.session_state.gender}\n"
        f"Weight: {st.session_state.weight} kg\n"
        f"Past Diseases: {st.session_state.past_diseases or 'None'}\n"
        f"[/USER_HEALTH_DETAILS]\n\n"
        f"[INTERNAL_KB]\n{kb_text}\n[/INTERNAL_KB]"
    )

    img_bytes = _image_bytes(st.session_state.img_disp)

    inputs = [
        {"text": context_text},
        {"inline_data": {"mime_type": "image/jpeg", "data": img_bytes}},
    ]

    with st.spinner("Analyzing image with patient details..."):
        result = _run_llm(PROMPT, context_text, inputs)

    st.session_state.safe_text = _scrub_medication_doses(result)


# =========================================================
# OUTPUT TABS
# =========================================================
tab_diag, tab_research, tab_down = st.tabs([
    "🩻 Diagnosis",
    "📚 Research Context",
    "⬇️ Downloads"
])


# ---------------- Diagnosis Tab ----------------
with tab_diag:
    if st.session_state.img_disp:
        st.image(st.session_state.img_disp, caption="Uploaded Image", use_container_width=True)

    if st.session_state.safe_text:
        st.markdown("### 🧾 Diagnostic Report")
        st.markdown(
            f"<div class='report-box'>{st.session_state.safe_text}</div>",
            unsafe_allow_html=True
        )
        st.info(DISCLAIMER)


# ---------------- Research Tab ----------------
with tab_research:
    st.markdown("### 🌐 Research Context")
    cond = _extract_condition_name(st.session_state.safe_text)
    if cond:
        st.markdown(f"#### Verified Research for **{cond}**")
        refs = _fetch_verified_references(cond, max_results=10)
        for r in refs:
            st.markdown(r, unsafe_allow_html=True)
    else:
        st.info("No condition detected to fetch research.")


# ---------------- Downloads Tab ----------------
with tab_down:
    if st.session_state.safe_text:

        st.markdown("### Download Report")

        c1, c2 = st.columns(2)

        with c1:
            if st.button("📄 Generate PDF"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                for line in st.session_state.safe_text.splitlines():
                    pdf.multi_cell(0, 8, line)
                pdf.output("Arogyam_Report.pdf")

                with open("Arogyam_Report.pdf", "rb") as f:
                    st.download_button(
                        "⬇️ Download PDF",
                        f,
                        file_name="Arogyam_Report.pdf"
                    )

        with c2:
            if st.button("🔊 Generate Audio Summary"):
                tts = gTTS(text=st.session_state.safe_text[:4000], lang="en")
                tts.save("Arogyam_Audio.mp3")

                with open("Arogyam_Audio.mp3", "rb") as f:
                    st.download_button(
                        "⬇️ Download Audio",
                        f,
                        file_name="Arogyam_Audio.mp3"
                    )
    else:
        st.info("Run the analysis to enable downloads.")
