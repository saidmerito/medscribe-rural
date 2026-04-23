# 🏥 MedScribe Rural

> **Offline AI-powered digitization of handwritten medical records for rural health centers**  
> Built with **Gemma 4 E4B** · Runs 100% offline · Gemma 4 Good Hackathon 2026

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Gemma](https://img.shields.io/badge/Gemma_4-E4B-green)
![Ollama](https://img.shields.io/badge/Ollama-local-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🌍 The Problem

Rural health centers across the Horn of Africa (Djibouti, Ethiopia, Somalia) rely on paper-based registers written in French, Arabic, and Somali. Critical epidemiological data — cholera cases, vaccination rates, maternal health — remains locked in paper, invisible to district health authorities.

**MedScribe Rural** digitizes these records using local AI, with zero internet dependency.

---

## ✨ Features

- 📸 **Photo → Structured Data**: Photograph a register page, get clean JSON records
- 🤖 **Gemma 4 E4B**: Local LLM corrects OCR errors, maps ICD-10 codes, enforces schema via function calling
- 🌐 **Multilingual**: French, Arabic, Somali, Afar
- 📊 **Auto Reports**: Weekly/monthly PDF and CSV epidemiological reports
- 🔌 **100% Offline**: No internet required — runs on a standard laptop
- 🔄 **Optional Sync**: Delta sync to central dashboard when connectivity available

---

## 🏗️ Architecture

```
[Photo of register]
        ↓
[Image preprocessing] → OpenCV (deskew, threshold, crop)
        ↓
[OCR] → PaddleOCR (multilingual: FR, AR, Somali)
        ↓
[Gemma 4 E4B via Ollama] → Error correction + structuring + ICD-10 coding
        ↓
[SQLite database] → Local storage
        ↓
[FastAPI backend] → REST API
        ↓
[Gradio UI] → Health worker interface + PDF/CSV report generation
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed
- 8GB RAM minimum

### 1. Clone and install

```bash
git clone https://github.com/yourusername/medscribe-rural.git
cd medscribe-rural
pip install -r requirements.txt
```

### 2. Pull Gemma 4 E4B

```bash
ollama pull gemma4:e4b
```

### 3. Run the app

```bash
python main.py
```

Open your browser at `http://localhost:7860`

---

## 📁 Project Structure

```
medscribe_rural/
├── main.py                 # Entry point
├── requirements.txt
├── config.py               # Configuration
├── app/
│   └── gradio_ui.py        # Gradio interface
├── ocr/
│   └── extractor.py        # PaddleOCR pipeline + image preprocessing
├── ai/
│   └── structurer.py       # Gemma 4 E4B via Ollama — core AI logic
├── db/
│   └── database.py         # SQLite operations
├── reports/
│   └── generator.py        # PDF + CSV report generation
├── utils/
│   └── image_utils.py      # Image preprocessing helpers
├── tests/
│   └── test_pipeline.py    # Unit tests
└── sample_data/
    └── sample_register.jpg # Example register image for demo
```

---

## 🧠 How Gemma 4 E4B Is Used

Gemma 4 E4B is the intelligence core of MedScribe Rural:

| Task | How Gemma 4 handles it |
|------|------------------------|
| OCR error correction | Reconstructs garbled medical terms using context |
| Multilingual normalization | Handles FR/AR/Somali mixed entries |
| ICD-10 coding | Maps plain-text diagnoses to standard codes |
| Schema enforcement | Native function calling guarantees valid JSON output |
| Confidence scoring | Flags low-confidence fields instead of hallucinating |

---

## 📊 Output Schema

```json
{
  "patient_id": "DJ-2024-00142",
  "date": "2024-03-15",
  "age": 7,
  "sex": "F",
  "chief_complaint": "fever and vomiting",
  "diagnosis": "Acute Watery Diarrhea",
  "icd10_code": "A09",
  "treatment": "ORS + Zinc",
  "outcome": "referred",
  "confidence": 0.94,
  "flags": []
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

Built for the **Gemma 4 Good Hackathon 2026** · Track: Health & Sciences  
Powered by Google's Gemma 4 open model family.
