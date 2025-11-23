# 🩺 AROGYAM - Agentic AI for Medical Imaging and Diagnosis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini%202.0-4285F4.svg)](https://ai.google.dev/)
[![CLIP](https://img.shields.io/badge/Model-CLIP--ViT--B--32-orange.svg)](https://github.com/openai/CLIP)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()


> **⚠️ DISCLAIMER**: This application is for **educational and research purposes only**. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Always consult certified healthcare professionals for medical concerns.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🌟 Overview

**AROGYAM** is an advanced AI-powered medical imaging analysis system that leverages state-of-the-art machine learning models to assist in medical diagnosis. The application combines multiple AI technologies including Google's Gemini 2.0 Flash for multimodal analysis, CLIP for image-text similarity validation, and RAG (Retrieval-Augmented Generation) for enhanced diagnostic accuracy.

### Key Highlights

- 🤖 **Agentic AI Architecture**: Multi-stage validation and analysis pipeline
- 🖼️ **Multi-Format Support**: JPEG, PNG, BMP, WebP, and DICOM medical images
- 🔍 **CLIP Validation**: Ensures relevance between user prompts and uploaded images
- 📚 **Knowledge Base Integration**: RAG-based retrieval from medical literature
- 🌐 **Research Context**: Automatic fetching of verified medical research papers
- 📄 **Report Generation**: PDF and audio summary exports
- 🎯 **Patient-Centric**: Incorporates patient demographics and medical history

## ✨ Features

### Core Functionality

1. **Medical Image Analysis**
   - Support for standard image formats (JPEG, PNG, BMP, WebP)
   - DICOM medical imaging format support
   - Automatic image preprocessing and orientation correction
   - High-resolution image handling (up to 10MB)

2. **AI-Powered Diagnosis**
   - Google Gemini 2.0 Flash multimodal analysis
   - Context-aware diagnosis using patient details
   - Integration with internal medical knowledge base
   - Medication dose scrubbing for safety compliance

3. **Validation Pipeline**
   - CLIP-based similarity scoring (threshold: 0.25)
   - Three-stage validation process:
     - Stage 1: Image-prompt relevance check
     - Stage 2: Patient details collection
     - Stage 3: Comprehensive diagnosis

4. **Research Integration**
   - Automatic condition extraction from diagnosis
   - Verified source fetching (WHO, NIH, PubMed, CDC, Mayo Clinic, etc.)
   - Real-time research paper retrieval
   - Citation-ready references

5. **Report Generation**
   - Professional PDF reports with FPDF
   - Audio summaries using Google Text-to-Speech
   - Downloadable formats for patient records

6. **Knowledge Base**
   - Vector-based semantic search using ChromaDB
   - Medical terminology database
   - Disease-specific guidelines
   - Radiology and dermatology knowledge bases

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (Streamlit)               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    CLIP Similarity Check                     │
│              (Image-Prompt Relevance Validation)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Patient Details Collection                  │
│         (Age, Gender, Weight, Medical History)               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Knowledge Base Retrieval (RAG)                  │
│           (ChromaDB Vector Store + Embeddings)               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            Gemini 2.0 Flash Multimodal Analysis              │
│        (Image + Text + Context → Diagnosis)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Post-Processing                           │
│         (Dose Scrubbing, Formatting, Safety Checks)          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Output Generation                           │
│         (Diagnosis Report, Research Links, Downloads)        │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Google API Key (for Gemini API)
- 4GB+ RAM recommended
- Internet connection for research fetching

### Step 1: Clone the Repository

```bash
git clone https://github.com/kartikpagariya25/Arogyam-Agentic-AI-for-medical-imaging-and-diagnosis.git
cd arogyam
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

To get a Google API key:
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy and paste it into your `.env` file

### Step 5: Initialize Knowledge Base

```bash
# The vector stores will be automatically created on first run
# Ensure the data/knowledge/ directory contains your medical documents
```

## 💻 Usage

### Running the Application

```bash
streamlit run main.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Application

1. **Select Mode**: Choose "Medical Image Analysis" from the sidebar
2. **Enter Prompt**: Describe your medical concern in the text area
3. **Add Symptoms** (Optional): Include additional symptoms
4. **Upload Image**: Select a medical image (JPEG, PNG, DICOM, etc.)
5. **Click Analyze**: Start the analysis process
6. **Fill Patient Details**: Enter age, gender, weight, and medical history
7. **View Results**: 
   - **Diagnosis Tab**: View the AI-generated diagnostic report
   - **Research Tab**: Explore verified medical research papers
   - **Downloads Tab**: Generate and download PDF/audio reports

### Example Use Cases

- **Dermatology**: Analyze skin conditions (psoriasis, athlete's foot, rashes)
- **Radiology**: Examine X-rays, CT scans, MRI images
- **Ophthalmology**: Assess eye conditions (conjunctivitis, etc.)
- **Orthopedics**: Evaluate bone fractures and injuries

## 📁 Project Structure

```
Arogyam/
│
├── main.py                      # Main Streamlit application
├── prompt_config.py             # System prompts for Gemini
├── report_prompt.py             # Report generation prompts
├── knowledge_base.py            # Text-based knowledge retrieval
├── image_knowledge_base.py      # Image-based knowledge retrieval
├── disease_translations.py      # Medical terminology translations
├── .env                         # Environment variables (not in repo)
├── .gitignore                   # Git ignore file
├── README.md                    # This file
│
├── data/
│   ├── knowledge/               # Medical knowledge documents
│   │   ├── medical_terms.txt
│   │   ├── psoriasis_guidelines.txt
│   │   ├── radiology_basics.txt
│   │   ├── skin_diseases.txt
│   │   └── conjuctiva1.pdf
│   │
│   └── skin_images/             # Sample medical images
│       └── athelete_foot/
│
├── Images/                      # Test images
│   ├── broken arm.jpg
│   ├── conjuctivitis.png
│   ├── psoriasis.jpg
│   └── rashes.webp
│
├── vector_store/                # ChromaDB text embeddings
│   └── chroma.sqlite3
│
└── vector_store_images/         # ChromaDB image embeddings
    └── chroma.sqlite3
```

## 🛠️ Technologies Used

### AI & Machine Learning
- **Google Gemini 2.0 Flash**: Multimodal AI for image and text analysis
- **CLIP (ViT-B-32)**: Image-text similarity validation
- **Sentence Transformers**: Semantic embeddings
- **ChromaDB**: Vector database for RAG

### Web Framework
- **Streamlit**: Interactive web application framework

### Medical Imaging
- **PyDICOM**: DICOM medical image format support
- **Pillow (PIL)**: Image processing and manipulation

### Document Processing
- **FPDF**: PDF report generation
- **gTTS**: Text-to-speech for audio reports

### Search & Retrieval
- **DuckDuckGo Search**: Verified medical research fetching

### Utilities
- **python-dotenv**: Environment variable management
- **NumPy**: Numerical computations
- **Requests**: HTTP requests for research validation

## ⚙️ Configuration

### Model Configuration

Edit `prompt_config.py` to customize the system prompts:

```python
PROMPT = """
Your custom system prompt here...
"""
```

### CLIP Threshold

Adjust the similarity threshold in `main.py`:

```python
CLIP_THRESHOLD = 0.25  # Range: 0.0 to 1.0
```

### Knowledge Base

Add medical documents to `data/knowledge/` directory. Supported formats:
- `.txt` - Plain text medical documents
- `.pdf` - PDF medical literature

### Verified Research Sources

Modify the verified sources list in `main.py`:

```python
verified = [
    "who.int", "nih.gov", "pubmed.ncbi.nlm.nih.gov",
    "cdc.gov", "mayoclinic.org", "clevelandclinic.org",
    # Add more trusted sources
]
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Notes

### Medical Disclaimer

- This tool is **NOT FDA approved**
- **NOT** a replacement for professional medical diagnosis
- Results should be **verified by licensed healthcare providers**
- Use only for **educational and research purposes**

### Privacy & Security

- Patient data is processed locally
- No data is stored on external servers (except API calls to Google)
- Ensure HIPAA compliance if used in clinical settings
- Implement proper data encryption for production use

### Limitations

- AI models can make mistakes
- Accuracy depends on image quality and prompt clarity
- Limited to conditions in the knowledge base
- Requires internet connection for research fetching


<div align="center">

**Made with ❤️ for the medical community**

⭐ Star this repo if you find it helpful!

</div>


