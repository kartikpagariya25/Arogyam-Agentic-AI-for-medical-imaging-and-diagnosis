<h1 align="center">🩺 AROGYAM</h1>
<p align="center">
  <b>Agentic AI for Medical Imaging and Diagnosis</b><br>
  Built with Streamlit • Google Gemini 2.0 Flash • ChromaDB • CLIP Embeddings
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white" alt="Python Badge"/>
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit Badge"/>
  <img src="https://img.shields.io/badge/Model-Google%20Gemini%202.0%20Flash-FFD700?logo=google&logoColor=black" alt="Gemini Badge"/>
  <img src="https://img.shields.io/badge/Database-ChromaDB-00C4CC?logo=databricks&logoColor=white" alt="ChromaDB Badge"/>
  <img src="https://img.shields.io/badge/Embeddings-CLIP%20(ViT--B--32)-800080?logo=openai&logoColor=white" alt="CLIP Badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?logo=github&logoColor=white" alt="License Badge"/>
</p>


## 🧠 Overview

Arogyam provides:
- AI-powered diagnosis for **images (X-ray, MRI, skin)** and **reports (PDF/TXT)**
- Structured and explainable reports with medical accuracy
- Verified reference links from **WHO, NIH, CDC, PubMed, Mayo Clinic**, and others
- Integrated **Text and Image Knowledge Bases**
- Dark-themed Streamlit interface with PDF and audio output options

---

## ✨ Features

- **Dual Diagnostic Modes:** Image or Report analysis  
- **AI-Powered Reports:** Uses Gemini 2.0 Flash for accurate medical insights  
- **Knowledge Bases:**  
  - Text KB with ChromaDB + Sentence Transformers  
  - Image KB with CLIP embeddings  
- **Verified References:** Auto-generated URLs from reliable health organizations  
- **Export Options:** PDF and audio summary generation  
- **User Interface:** Built on Streamlit (dark theme with multi-tab layout)

---

## 🧩 System Architecture

```mermaid
flowchart TD
    A["User Uploads Image or Report"] --> B["Input Validation & Similarity Check - CLIP"]
    B --> C["Gemini 2.0 Flash Model"]
    C --> D["Structured Diagnostic Report Generation"]
    D --> E["Extract Research Context"]
    D --> F["Retrieve Knowledge Base Context - ChromaDB"]
    E --> G["Display Verified Medical References"]
    F --> G
    G --> H["Generate PDF and Audio Outputs"]
```

## 🧰 Technology Stack

| Component | Technology |
|------------|-------------|
| Frontend UI | Streamlit |
| Core AI Model | Google Gemini 2.0 Flash |
| Text Knowledge Base | ChromaDB + all-MiniLM-L6-v2 |
| Image Knowledge Base | CLIP (ViT-B-32) |
| Search API | DuckDuckGo Search |
| PDF Generation | FPDF |
| Audio Generation | gTTS |
| Language | Python 3.10 |


## 📂 Project Structure
```
Arogyam/
├── main.py                   # Streamlit App (Diagnosis | Research Context | Downloads)
├── prompt_config.py           # Diagnostic prompt configuration
├── report_prompt.py           # Report analysis prompt
├── knowledge_base.py          # Text Knowledge Base
├── image_knowledge_base.py    # Image Knowledge Base
├── disease_translations.py    # Multilingual disease name support
│
├── data/
│   ├── knowledge/             # Medical text/PDF data
│   └── skin_images/           # Image dataset
│
├── vector_store_text/         # Text embeddings
├── vector_store_images/       # Image embeddings
│
├── .env                       # API Key
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup
```
1️⃣ Prerequisites
Install Python 3.10+
Create a free Google API Key from Google AI Studio
Verify installations:
python --version
pip --version
```

```
2️⃣ Clone the Repository
git clone https://github.com/kartikpagariya25/Arogyam-Agentic-AI-for-medical-imaging-and-diagnosis.git
cd Arogyam-Agentic-AI-for-medical-imaging-and-diagnosis
```

```
3️⃣ Create and Activate Virtual Environment
python -m venv arogyaamai
Activate it:
Windows:
arogyaamai\Scripts\activate

macOS/Linux:
source arogyaamai/bin/activate
```
```
4️⃣ Install Dependencies

pip install -r requirements.txt

If requirements.txt is missing, install manually:
pip install streamlit google-generativeai chromadb sentence-transformers duckduckgo-search pillow numpy fpdf gtts python-dotenv pydicom
```

```
5️⃣ Configure API Key
Create a .env file in the project root:
GOOGLE_API_KEY=your_gemini_api_key_here
```

```
6️⃣ Build Knowledge Bases
a. Text Knowledge Base
python knowledge_base.py --build

b. Image Knowledge Base
python image_knowledge_base.py --build

You should see:
[indexed] image_name.jpg
[done] Image KB built.
```

```
7️⃣ Run the Application
streamlit run main.py
Visit your browser:
http://localhost:8501
💡 How to Use
1) Select analysis mode — Medical Image or Health Report.
2) Upload your file (.jpg, .png, .pdf, .txt, .dcm).
3) Enter your diagnostic prompt (e.g., “check for psoriasis”).
4) Click Analyze.
5) Navigate through:
    Diagnosis Tab: Full AI-generated medical report
    Research Context Tab: Verified health references
    Downloads Tab: Export report (PDF/Audio)

🧬 Example Output
Detected Condition: Psoriasis
References:
WHO on Psoriasis
NIH: Psoriasis
CDC Info
PubMed Articles
Mayo Clinic – Psoriasis
```

## 👥 Team Members (2025 Batch — AIDS, VIT Pune)

| Name | Role | Responsibilities |
|------|------|------------------|
| **Kartik Pagariya** | Project Lead / AI Developer | Gemini integration, diagnostic logic, prompt design |
| **Aditya Dengale** | Data Engineer | Knowledge Base creation, embedding retrieval |
| **Pooja Wavdara** | Frontend Developer | Streamlit UI, CSS theme, download system |
| **Prajakta Patil** | Backend & Validation | Research Context logic, similarity check, testing |

## 🛠 Troubleshooting

| Issue | Cause | Solution |
|--------|--------|-----------|
| `GOOGLE_API_KEY not set` | `.env` missing or invalid | Add valid key to `.env` |
| `ModuleNotFoundError` | Dependency missing | Run `pip install -r requirements.txt` |
| `Streamlit port conflict` | Another app using port 8501 | Run `streamlit run main.py --server.port 8502` |
| `ChromaDB Error` | Version mismatch | Run `pip install --upgrade chromadb` |


## ⚠️ Disclaimer
```
Arogyam is a research and educational tool only.
It is not a medical diagnostic device.
Always consult certified healthcare professionals for medical advice.
```

🧾 License
Distributed under the MIT License.
You are free to use, modify, and distribute with attribution.
