# config.py
import os

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma4:e4b")

# Database
DB_PATH = os.getenv("DB_PATH", "medscribe.db")

# OCR
OCR_LANGUAGES = ["fr", "ar", "en"]  # French, Arabic, English (covers Somali in Latin script)
OCR_USE_GPU = False  # Edge deployment: CPU only

# Reports output directory
REPORTS_DIR = os.getenv("REPORTS_DIR", "reports_output")

# Confidence threshold below which fields are flagged
CONFIDENCE_THRESHOLD = 0.75

# Epidemic alert thresholds (cases per week per facility)
EPIDEMIC_THRESHOLDS = {
    "A00": 2,   # Cholera — any case is an alert
    "A09": 10,  # Acute watery diarrhea
    "B54": 15,  # Malaria
    "A15": 3,   # Tuberculosis
}
