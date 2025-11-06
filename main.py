# main.py — AROGYAM (with Research Context integration)

import os
import io
import re
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="AROGYAM — Agentic AI for Medical Imaging and Diagnosis", layout="wide")

from PIL import Image as PILImage, ImageOps as PILImageOps
import numpy as np
from dotenv import load_dotenv

try:
    import pydicom
except Exception:
    pydicom = None

import google.generativeai as genai
from duckduckgo_search import DDGS
from prompt_config import PROMPT
from knowledge_base import search_kb
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# Constants
# -----------------------------
APP_TITLE = "AROGYAM — Agentic AI for Medical Imaging and Diagnosis"
DISCLAIMER = (
    "This analysis is for educational and informational purposes only. "
    "It is not a medical diagnosis or prescription. "
    "Always consult a qualified healthcare professional for any medical concerns."
)
MAX_IMAGE_MB = 10
MODEL_NAME = "models/gemini-2.0-flash"

# -----------------------------
# Utility functions
# -----------------------------
def _scrub_medication_doses(text: str) -> str:
    if not text:
        return text
    pattern = re.compile(r"(\b\d+(\.\d+)?\s*(mg|mcg|g|gram|ml|mL|units|IU)\b[^.\n]*)", re.I)
    return pattern.sub("[dose removed — clinician required]", text)


def _load_image_any(file: io.BytesIO, filename: str) -> PILImage.Image:
    suffix = Path(filename).suffix.lower()
    if suffix == ".dcm":
        if pydicom is None:
            raise RuntimeError("pydicom not installed. Cannot read DICOM.")
        ds = pydicom.dcmread(file)
        arr = ds.pixel_array.astype(np.float32)
        arr = 255 * (arr - arr.min()) / (arr.ptp() or 1)
        arr8 = arr.astype(np.uint8)
        img = PILImage.fromarray(arr8, mode="L") if arr8.ndim == 2 else PILImage.fromarray(arr8[..., :3])
    else:
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
    """Load CLIP model once for consistency checking."""
    return SentenceTransformer("clip-ViT-B-32")


def _check_prompt_image_similarity(prompt_text: str, image: PILImage.Image) -> bool:
    """Return True if the prompt and image are semantically related."""
    model = _get_clip_model()
    text_emb = model.encode([prompt_text], convert_to_tensor=True)
    image_emb = model.encode([image], convert_to_tensor=True)
    score = util.cos_sim(text_emb, image_emb).item()
    return score >= 0.25  # threshold can be tuned


@st.cache_data(show_spinner=False)
def _run_llm_analysis(system_prompt: str, context_text: str, image_bytes: bytes) -> str:
    model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_prompt)
    result = model.generate_content(
        contents=[
            {"text": context_text},
            {"inline_data": {"mime_type": "image/jpeg", "data": image_bytes}},
        ],
        safety_settings=None,
        generation_config={"temperature": 0.25, "top_p": 0.9},
    )
    return result.text or ""

# -----------------------------
# NEW: Web Research Retrieval
# -----------------------------
def _get_research_links(query: str, max_links: int = 5) -> list[str]:
    """Fetch top trusted medical research links from WHO, NIH, PubMed, CDC, etc."""
    sources = ["who.int", "nih.gov", "pubmed.ncbi.nlm.nih.gov", "cdc.gov", "mayoclinic.org"]
    ddg = DDGS()
    results = []
    for site in sources:
        try:
            search_results = ddg.text(f"{query} site:{site}", max_results=2)
            for r in search_results:
                results.append(f"- [{r['title']}]({r['href']}) — {site}")
                if len(results) >= max_links:
                    break
        except Exception:
            continue
    return results[:max_links]

# -----------------------------
# Streamlit UI
# -----------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY", "")
if not api_key:
    st.warning("GOOGLE_API_KEY is not set. Add it to your environment or .env file.")
else:
    genai.configure(api_key=api_key)

st.title(APP_TITLE)
st.caption(DISCLAIMER)

with st.sidebar:
    user_prompt = st.text_area("Your Prompt", key="prompt_area")
    uploaded = st.file_uploader(
        "Upload Medical Image (.jpg, .jpeg, .png, .bmp, .webp, .dcm)",
        type=["jpg", "jpeg", "png", "bmp", "webp", "dcm"],
        key="file_uploader",
    )
    run_btn = st.button("Analyze", key="analyze_button")

if uploaded is not None:
    try:
        img_preview = _load_image_any(io.BytesIO(uploaded.getvalue()), uploaded.name)
        img_preview = _resize_for_display(img_preview)
        st.image(img_preview, caption="Uploaded Image", use_container_width=True)
    except Exception as e:
        st.error(f"Unable to display image: {e}")

# -----------------------------
# Execution
# -----------------------------
if run_btn:
    if not api_key:
        st.error("Missing GOOGLE_API_KEY.")
        st.stop()

    if not user_prompt.strip():
        st.error("Enter a valid prompt first.")
        st.stop()

    if uploaded is None:
        st.error("Upload an image first.")
        st.stop()

    size_mb = len(uploaded.getvalue()) / (1024 * 1024)
    if size_mb > MAX_IMAGE_MB:
        st.error(f"File size {size_mb:.1f} MB exceeds {MAX_IMAGE_MB} MB limit.")
        st.stop()

    try:
        img = _load_image_any(io.BytesIO(uploaded.getvalue()), uploaded.name)
        img_disp = _resize_for_display(img)
        img_bytes = _image_bytes(img_disp)
    except Exception as e:
        st.error(f"Image error: {e}")
        st.stop()

    # --- CLIP Consistency Check ---
    with st.spinner("Verifying prompt and image consistency..."):
        try:
            if not _check_prompt_image_similarity(user_prompt, img_disp):
                st.warning(
                    "The uploaded image and entered prompt appear unrelated.\n"
                    "Please upload an image relevant to your described condition."
                )
                st.stop()
        except Exception as e:
            st.error(f"Error in similarity check: {e}")
            st.stop()

    # --- Construct Base Context ---
    context_text = (
        f"User Prompt: {user_prompt}\n\n"
        "Follow the diagnostic and reporting steps from the system prompt precisely."
    )

    # --- Knowledge Base Integration ---
    kb_refs = search_kb(user_prompt, top_k=5)
    if kb_refs:
        kb_context = "\n\n".join(kb_refs)
        context_text += (
            "\n\n### Reference Context (from internal Medical Knowledge Base)\n"
            f"{kb_context}\n\n"
            "Use this only as background information. "
            "Do not infer conditions not visible in the image."
        )

    # --- NEW: Research Context (Trusted Web References) ---
    with st.spinner("Fetching trusted medical research references..."):
        research_links = _get_research_links(user_prompt)
        if research_links:
            research_section = "\n".join(research_links)
            context_text += (
                "\n\n### Research Context (trusted web references)\n"
                f"{research_section}\n\n"
                "These links provide verified background from health organizations."
            )

    # --- Run Gemini Model ---
    with st.status("Analyzing with Gemini 2.0 Flash…", expanded=False) as status:
        try:
            result_text = _run_llm_analysis(PROMPT, context_text, img_bytes)
        except Exception as e:
            st.error(f"Model error: {e}")
            st.stop()

        safe_text = _scrub_medication_doses(result_text)
        status.update(label="Analysis complete.", state="complete")

    # --- Output Display ---
    st.subheader("Detailed Diagnostic Report")
    st.markdown(safe_text)
    st.info(DISCLAIMER)

    if research_links:
        st.subheader("Research Context — Verified Medical References")
        st.markdown("\n".join(research_links))
