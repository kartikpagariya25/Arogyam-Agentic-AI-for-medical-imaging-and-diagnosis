# AI Agent Instructions for Radiology Analysis Project

This project is a medical imaging analysis system that provides detailed reports with translations in Hindi and Marathi. Here's what you need to know to work effectively with this codebase.

## 🏗️ Architecture Overview

- **Main Components**:
  - `main.py`: Core application with image analysis, translation, and PDF generation
  - `disease_translations.py`: Dictionary of disease translations (English -> Hindi/Marathi)
  - `prompt_config.py`: Structured prompt template for medical image analysis
  - `fonts/`: Contains Noto Sans Devanagari font for PDF generation

## 🔑 Key Integration Points

1. **AI Integration**:
   - Uses Gemini model via the `agno` library for image analysis and translations
   - Two separate agents: `medical_agent` for analysis and `translator_agent` for translations
   - Requires `GOOGLE_API_KEY` in environment variables

2. **Image Processing**:
   - Uses PIL for image handling and resizing
   - Default resizing to 1000px width while maintaining aspect ratio
   - Support for keeping full resolution via `keep_full_resolution` parameter

3. **External Dependencies**:
   - DuckDuckGo integration for research context in medical reports
   - FPDF for PDF report generation with multi-language support
   - gTTS (Google Text-to-Speech) for audio generation

## 📋 Project-Specific Conventions

1. **Report Structure**:
   - All medical reports follow a strict 7-section format (see `PROMPT` in `prompt_config.py`)
   - Each section requires minimum content lengths and specific formatting
   - Reports must be 500-800 words minimum

2. **Translation Patterns**:
   - Disease translations are first looked up in `DISEASE_TRANSLATIONS`
   - Fallback to dynamic translation via translator_agent
   - Format: "English (Hindi / Marathi)"

3. **Error Handling**:
   - Image analysis errors are logged and returned with "⚠️ Analysis error" prefix
   - Translation failures gracefully fall back to English-only names
   - Missing font files fallback to Arial

## 💡 Development Workflow

1. **Environment Setup**:
   ```powershell
   # Create .env file with API key
   echo "GOOGLE_API_KEY=your-key-here" > .env
   ```

2. **Testing Images**:
   - Test images should be placed in the `Images/` directory
   - Supported formats: Common image formats (PNG, JPEG, etc.)
   - Consider both normal and high-resolution test cases

3. **PDF Generation**:
   - Ensure Noto Sans Devanagari font is present in `fonts/` for proper Hindi/Marathi rendering
   - PDF generation automatically handles multilingual content

## 🎯 Common Tasks

1. **Adding New Disease Translations**:
   - Add entries to `DISEASE_TRANSLATIONS` dictionary in `disease_translations.py`
   - Follow format: `"english_name": "marathi_name (Marathi) / hindi_name (Hindi)"`

2. **Modifying Report Structure**:
   - Update the `PROMPT` constant in `prompt_config.py`
   - Maintain markdown formatting and section numbering
   - Ensure minimum content requirements are specified