"""
Streamlit UI for the RAG Q&A pipeline.
Provides a minimal chat-like interface for asking questions
and an evaluation panel for testing answer quality.
"""

import os
import streamlit as st
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def get_health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json()
    except Exception:
        return None

# --- Page Config ---
st.set_page_config(page_title="RAG Q&A", page_icon="🔍", layout="wide")
st.title("🔍 RAG Q&A Pipeline")