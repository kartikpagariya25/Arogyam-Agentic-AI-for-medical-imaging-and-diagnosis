# main.py — AROGYAM (Stable Release with CLIP Validation & Advisory Note)

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
except Exception:
    pydicom = None

# ---------------------------------------------------------
# Streamlit Setup & Styling
# ---------------------------------------------------------
st.set_page_config(page_title="AROGYAM — Agentic AI for Medical Imaging and Diagnosis", layout="wide")

st.markdown("""
<style>
body { background-color:#0E1117; color:#FAFAFA; }
section[data-testid="stSidebar"] {
  background-color:#111827 !important; color:#FAFAFA !important;
  border-right:1px solid #222;
}
.sidebar-title{font-weight:700;font-size:18px;color:#E5E7EB;margin-bottom:0.75rem;}
label,span,p{color:#E5E7EB !important;}
textarea,input,div[data-baseweb="input"]>div{
  background-color:#1F2937 !important;color:#FAFAFA !important;
  border:1px solid #374151 !important;border-radius:6px !important;
}
div.stButton>button{
  background-color:#2563EB !important;color:#fff !important;border:none !important;
  border-radius:8px !important;font-weight:600 !important;height:2.6rem !important;width:100%;
}
div.stButton>button:hover{background-color:#1E40AF !important;}
.report-box{background-color:#1F2937;border-radius:12px;padding:20px;color:#E5E7EB;
  border:1px solid #374151;box-shadow:0 2px 4px rgba(0,0,0,0.4);}
h1,h2,h3{color:#E5E7EB;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
APP_TITLE = "🩺 AROGYAM — Agentic AI for Medical Imaging and Diagnosis"
DISCLAIMER = (
    "⚠️ This report is for educational use only. It is not a medical prescription. "
    "Consult certified healthcare professionals for diagnosis or treatment."
)
MODEL_NAME = "models/gemini-2.0-flash"
CLIP_THRESHOLD = 0.30       # balanced threshold
MAX_IMAGE_MB = 10

# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------
def _scrub_medication_doses(text: str) -> str:
    if not text:
        return text
    pat = re.compile(r"(\b\d+(\.\d+)?\s*(mg|mcg|g|gram|ml|mL|units|IU)\b[^.\n]*)", re.I)
    return pat.sub("[dose removed — clinician required]", text)


def _extract_condition_name(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"(?i)diagnosis summary[:\-]?\s*([A-Za-z][A-Za-z \-/]+)", text)
    if m: return m.group(1).strip()
    m = re.search(r"\b([A-Z][a-z]+(?:itis|osis|emia|opathy|oma|algia|dermatitis|psoriasis|asthma|pneumonia))\b", text)
    return m.group(1).strip() if m else ""


@st.cache_data(ttl=86400)
def _fetch_verified_references(disease_name: str, max_results: int = 10):
    query = (
        f'"{disease_name}" (research OR study OR article OR review OR clinical trial) '
        f'site:who.int OR site:nih.gov OR site:pubmed.ncbi.nlm.nih.gov OR site:cdc.gov '
        f'OR site:mayoclinic.org OR site:clevelandclinic.org OR site:nature.com '
        f'OR site:springer.com OR site:thelancet.com OR site:nejm.org'
    )
    verified_domains = [
        "who.int", "nih.gov", "pubmed.ncbi.nlm.nih.gov", "cdc.gov",
        "mayoclinic.org", "clevelandclinic.org", "nature.com",
        "springer.com", "thelancet.com", "nejm.org"
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
            if not any(domain.endswith(v) for v in verified_domains):
                continue
            try:
                resp = requests.get(url, timeout=5, stream=True)
                if not (200 <= resp.status_code < 300):
                    continue
            except Exception:
                continue
            results.append(f"**[{title}]({url})**  \n_{snippet}_")
            if len(results) >= max_results:
                break
    if not results:
        fallback = {
            "WHO": f"https://www.who.int/search?q={disease_name}",
            "NIH": f"https://www.nih.gov/search?query={disease_name}",
            "PubMed": f"https://pubmed.ncbi.nlm.nih.gov/?term={disease_name}",
            "CDC": f"https://www.cdc.gov/search/?query={disease_name}",
            "Mayo Clinic": f"https://www.mayoclinic.org/search/search-results?q={disease_name}",
        }
        results = [f"**[{n}]({u})**" for n,u in fallback.items()]
    return results


def _load_image_any(file: io.BytesIO, filename: str) -> PILImage.Image:
    ext = Path(filename).suffix.lower()
    if ext == ".dcm":
        if pydicom is None:
            raise RuntimeError("pydicom not installed.")
        ds = pydicom.dcmread(file)
        arr = ds.pixel_array.astype(np.float32)
        arr = 255*(arr-arr.min())/(arr.ptp() or 1)
        return PILImage.fromarray(arr.astype(np.uint8))
    img = PILImage.open(file)
    img = PILImageOps.exif_transpose(img)
    return img.convert("RGB") if img.mode not in ("RGB","L") else img


def _resize_for_display(img: PILImage.Image, max_side: int = 1024) -> PILImage.Image:
    w,h = img.size
    s = min(max_side/max(w,h),1.0)
    return img.resize((int(w*s), int(h*s)), PILImage.LANCZOS) if s<1.0 else img


def _image_bytes(img: PILImage.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=90)
    return b.getvalue()


@st.cache_resource
def _get_clip_model():
    return SentenceTransformer("clip-ViT-B-32")


def _check_prompt_image_similarity(prompt_text: str, image: PILImage.Image) -> float:
    model = _get_clip_model()
    te = model.encode([prompt_text], convert_to_tensor=True, show_progress_bar=False)
    ie = model.encode([image],           convert_to_tensor=True, show_progress_bar=False)
    return float(util.cos_sim(te, ie).item())


@st.cache_data(show_spinner=False)
def _run_llm(system_prompt: str, context_text: str, inputs: list):
    model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_prompt)
    res = model.generate_content(contents=inputs, generation_config={"temperature":0.25,"top_p":0.9})
    return res.text or ""

# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY","")
if not api_key:
    st.error("GOOGLE_API_KEY not set.")
else:
    genai.configure(api_key=api_key)

st.markdown(f"## {APP_TITLE}")
st.caption(DISCLAIMER)

# ---------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------
st.sidebar.markdown("<p class='sidebar-title'>⚙️ Mode</p>", unsafe_allow_html=True)
mode = st.sidebar.radio("", ["Medical Image Analysis","Health Report Analysis"])

st.sidebar.markdown("<p class='sidebar-title'>⚠️ Important</p>", unsafe_allow_html=True)
st.sidebar.info("If the uploaded image and your written prompt are not related, "
                "the system may produce a **false or misleading diagnosis**.")

st.sidebar.markdown("<p class='sidebar-title'>🧠 Your Prompt</p>", unsafe_allow_html=True)
user_prompt = st.sidebar.text_area("Describe your concern or question", height=90)

st.sidebar.markdown("<p class='sidebar-title'>➕ Additional Symptoms</p>", unsafe_allow_html=True)
extra_symptoms = st.sidebar.text_area(
    "Optional. Used for reasoning only (ignored for CLIP validation).",
    height=80,
    placeholder="e.g., mild fever, rashes on lower back..."
)

with st.sidebar.expander("⚙️ Advanced"):
    if st.button("Clear cached references"):
        _fetch_verified_references.clear()
        st.success("Cleared cached research links.")

uploaded = st.sidebar.file_uploader(
    "📁 Upload File", type=["jpg","jpeg","png","bmp","webp","dcm","pdf","txt"]
)
run_btn = st.sidebar.button("🔍 Analyze")

tab_diag, tab_research, tab_down = st.tabs(["🩻 Diagnosis","📚 Research Context","⬇️ Downloads"])

if "safe_text" not in st.session_state:
    st.session_state.safe_text = ""
if "img_disp" not in st.session_state:
    st.session_state.img_disp = None

# ---------------------------------------------------------
# Analysis
# ---------------------------------------------------------
if run_btn:
    if not api_key: st.stop()
    if not user_prompt.strip():
        st.error("Enter a valid prompt."); st.stop()
    if uploaded is None:
        st.error("Upload a valid file."); st.stop()
    if uploaded.size > MAX_IMAGE_MB*1024*1024:
        st.error(f"File exceeds {MAX_IMAGE_MB} MB limit."); st.stop()

    kb_refs = search_kb(user_prompt, top_k=3) or []
    kb_text = "\n".join(kb_refs)
    sym_text = f"\n\nAdditional Symptoms:\n{extra_symptoms.strip()}" if extra_symptoms.strip() else ""
    context_text = f"User Prompt: {user_prompt}{sym_text}\n\n[INTERNAL_KB]\n{kb_text}\n[/INTERNAL_KB]"

    suffix = Path(uploaded.name).suffix.lower()
    if suffix in [".jpg",".jpeg",".png",".bmp",".webp",".dcm"]:
        img = _load_image_any(io.BytesIO(uploaded.getvalue()), uploaded.name)
        img_disp = _resize_for_display(img)
        with st.spinner("Validating image vs prompt..."):
            clip_score = _check_prompt_image_similarity(user_prompt, img_disp)
        st.caption(f"CLIP similarity: {clip_score:.3f} (required ≥ {CLIP_THRESHOLD})")
        if clip_score < CLIP_THRESHOLD:
            st.warning("Low CLIP match — image and prompt may be unrelated. "
                       "Diagnosis could be inaccurate (educational use only).")
        else:
            st.success("Prompt–image alignment validated.")
        img_bytes = _image_bytes(img_disp)
        inputs = [{"text":context_text},{"inline_data":{"mime_type":"image/jpeg","data":img_bytes}}]
        with st.spinner("Analyzing image..."):
            result_text = _run_llm(PROMPT, context_text, inputs)
        st.session_state.img_disp = img_disp
    else:
        if suffix == ".pdf":
            inputs = [{"text":context_text},{"inline_data":{"mime_type":"application/pdf","data":uploaded.getvalue()}}]
        else:
            report_text = uploaded.read().decode("utf-8",errors="ignore")
            context_text += "\n\nReport Text:\n"+report_text
            inputs = [{"text":context_text}]
        with st.spinner("Analyzing report..."):
            result_text = _run_llm(REPORT_PROMPT, context_text, inputs)

    st.session_state.safe_text = _scrub_medication_doses(result_text)

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
with tab_diag:
    if st.session_state.img_disp:
        st.image(st.session_state.img_disp, caption="Uploaded Image", use_container_width=True)
    if st.session_state.safe_text:
        st.markdown("### 🧾 Diagnostic Report")
        st.markdown(f"<div class='report-box'>{st.session_state.safe_text}</div>", unsafe_allow_html=True)
        st.info(DISCLAIMER)

with tab_research:
    st.markdown("### 🌐 Research Context — Trusted Medical References & Research Papers")
    cond = _extract_condition_name(st.session_state.safe_text)
    if cond:
        st.markdown(f"#### Curated sources for **{cond}**")
        refs = _fetch_verified_references(cond, max_results=10)
        for r in refs: st.markdown(r, unsafe_allow_html=True)
        st.caption("Verified live research links (WHO, NIH, PubMed, Mayo, Cleveland Clinic, Nature, Springer, etc.).")
    else:
        st.info("Run an analysis to fetch verified references.")

with tab_down:
    if st.session_state.safe_text:
        st.markdown("### 📂 Generate Outputs")
        c1,c2 = st.columns(2)
        with c1:
            if st.button("📄 Generate PDF Report"):
                pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12)
                for line in st.session_state.safe_text.splitlines():
                    pdf.multi_cell(0,8,line if line.strip() else "")
                pdf.output("Arogyam_Report.pdf")
                with open("Arogyam_Report.pdf","rb") as f:
                    st.download_button("⬇️ Download PDF", f, file_name="Arogyam_Report.pdf")
        with c2:
            if st.button("🔊 Generate Audio Summary"):
                tts = gTTS(text=st.session_state.safe_text[:4000], lang="en")
                tts.save("Arogyam_Audio.mp3")
                with open("Arogyam_Audio.mp3","rb") as f:
                    st.download_button("⬇️ Download Audio", f, file_name="Arogyam_Audio.mp3")
    else:
        st.info("Run an analysis to enable downloads.")
