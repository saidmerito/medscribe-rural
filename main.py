# main.py
"""
MedScribe Rural — Entry point
Launches the Gradio UI with the FastAPI backend.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

from app.gradio_ui import build_ui
from db.database import init_db

# Initialize database
init_db()

# FastAPI app (for optional REST API access / future DHIS2 sync)
api = FastAPI(title="MedScribe Rural API", version="1.0.0")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@api.get("/health")
def health_check():
    return {"status": "ok", "model": "gemma4:e4b", "mode": "offline"}

# Mount Gradio UI
demo = build_ui()
app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    print("\n🏥 MedScribe Rural starting...")
    print("📡 Make sure Ollama is running: ollama serve")
    print("🤖 Make sure Gemma 4 is pulled: ollama pull gemma4:e4b")
    print("🌐 Opening at http://localhost:7860\n")
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
