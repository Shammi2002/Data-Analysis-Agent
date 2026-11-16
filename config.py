"""Paths and limits shared by the whole app."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_datasets")
IMAGE_DIR = os.path.join(BASE_DIR, "images", "plotly_figures", "pickle")
PDF_DIR = os.path.join(BASE_DIR, "generated_pdfs")

for _folder in (UPLOAD_DIR, IMAGE_DIR, PDF_DIR):
    os.makedirs(_folder, exist_ok=True)

# Model
MODEL_NAME = "openai/gpt-oss-120b"  # Groq retired llama-3.3-70b-versatile on 16 Aug 2026
TEMPERATURE = 0.2
MAX_TOKENS = 4096
REASONING_EFFORT = "low"  # keeps answers fast and leaves room for the code

# Agent limits
RECURSION_LIMIT = 15        # max agent<->tools steps per question
MAX_HISTORY_MESSAGES = 8    # roughly the last 6 user+assistant exchanges
MAX_TOOL_OUTPUT = 800       # characters of tool output sent back to the model
MIN_PDF_BYTES = 1024        # smaller PDF files are treated as broken
