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

def ask_question(question: str) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        st.error(f"API error: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None

def evaluate_answer(question: str, reference: str) -> dict | None:
    try:
        r = requests.post(
            f"{API_URL}/evaluate",
            json={"question": question, "reference_answer": reference},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        st.error(f"API error: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None

def load_eval_example(question: str, reference: str) -> None:
    """Populate evaluation inputs from a preset example."""
    st.session_state["eval_q"] = question
    st.session_state["eval_ref"] = reference

# --- Page Config ---
st.set_page_config(page_title="RAG Q&A", page_icon="💡", layout="wide")
st.title("💡 RAG Q&A Pipeline")

# --- Sidebar ---
with st.sidebar:
    st.header("System Status")
    health = get_health()
    if health:
        st.success(f"API: Online")
    else:
        st.error("API: Offline — start the backend with `uvicorn app.api:app`")



# --- Main Tabs ---
tab_qa, tab_eval, tab_info = st.tabs(["💬 Ask Questions", "📊 Evaluate Answers", "🤓 Some info"])

# --- Q&A Tab ---
with tab_qa:
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the capital of Hungary?",
    )

    if st.button("Ask", type="primary", key="ask_btn"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                result = ask_question(question)
            if result:
                st.subheader("Answer")
                st.write(result["answer"])

                st.subheader("Sources")
                for i, source in enumerate(result["sources"], 1):
                    with st.expander(f"[{i}] {source['title']} (score: {source['score']:.4f})"):
                        st.write(source["text"])
                        if source.get("url"):
                            st.caption(f"Source: {source['url']}")

# --- Evaluation Tab ---
with tab_eval:
    st.markdown("""
    **Evaluate answer quality** by providing a question and a known reference answer.
    The system will generate an answer and compute quality metrics.
    """)

    eval_question = st.text_input(
        "Question:",
        placeholder="e.g., What is photosynthesis?",
        key="eval_q",
    )
    reference_answer = st.text_area(
        "Reference answer:",
        placeholder="Provide the expected / gold-standard answer here",
        key="eval_ref",
    )

    # Pre-built evaluation examples
    st.markdown("**Or try a pre-built example:**")
    examples = [
        {
            "question": "Who was the first king of Hungary?",
            "reference": "Stephen I was the first king of Hungary."
        },
        {
            "question": "What is the national language of Hungary?",
            "reference": "The national language of Hungary is Hungarian."
        },
        {
            "question": "What is Hungary's traditional currency?",
            "reference": "The traditional currency of Hungary is the Hungarian Forint."
        },
        {
            "question": "What is the largest thermal bath in Budapest?",
            "reference": "Széchenyi Thermal Bath is the largest thermal bath in Budapest."
        }
    ]
    for ex in examples:
        st.button(
            f"📝 {ex['question']}",
            key=f"ex_{ex['question'][:20]}",
            on_click=load_eval_example,
            args=(ex["question"], ex["reference"]),
        )

    if st.button("Evaluate", type="primary", key="eval_btn"):
        if not eval_question.strip() or not reference_answer.strip():
            st.warning("Please fill in both the question and reference answer.")
        else:
            with st.spinner("Generating and evaluating..."):
                result = evaluate_answer(eval_question, reference_answer)
            if result:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Generated Answer")
                    st.write(result["generated_answer"])
                with col2:
                    st.subheader("Reference Answer")
                    st.write(result["reference_answer"])

                st.subheader("Quality Metrics")
                metrics = result["metrics"]
                cols = st.columns(5)
                cols[0].metric("ROUGE-1", f"{metrics['rouge1']:.4f}")
                cols[1].metric("ROUGE-2", f"{metrics['rouge2']:.4f}")
                cols[2].metric("ROUGE-L", f"{metrics['rougeL']:.4f}")
                cols[3].metric("Semantic Sim.", f"{metrics['semantic_similarity']:.4f}")
                cols[4].metric("Faithfulness", f"{metrics['faithfulness_score']:.4f}")

                st.subheader("Retrieved Sources")
                for i, source in enumerate(result["sources"], 1):
                    with st.expander(f"[{i}] {source['title']} (score: {source['score']:.4f})"):
                        st.write(source["text"])

with tab_info:
    st.markdown("""
        # 🧠 RAG Q&A App

        This app answers your questions using **Simple English Wikipedia** as a knowledge base.  
        It uses a **Retrieval-Augmented Generation (RAG)** pipeline:

        1. **Retrieve** relevant passages from Wikipedia using vector search.
        2. **Re-rank** them for precision with a cross-encoder.
        3. **Generate** a fluent answer using a pretrained language model.

        All processing runs **locally**, no paid APIs needed.  
        You can ingest Wikipedia data, ask questions, and even evaluate answer quality, all in one interface.
    """)