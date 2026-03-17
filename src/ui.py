"""
Streamlit UI for the RAG Q&A pipeline.
Provides a minimal chat-like interface for asking questions
and an evaluation panel for testing answer quality.
"""

import logging

import streamlit as st

from src.rag_orchestrator import RAG_Orchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


st.set_page_config(page_title="RAG Q&A", page_icon="💡", layout="centered")
st.title('RAG Q&A')

@st.cache_resource
def get_orchestrator():
    return RAG_Orchestrator()

orchestrator = get_orchestrator()

def ask_question(question: str) -> dict | None:
    try:
        return orchestrator.answer(question)
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def evaluate_answer(question: str, reference: str) -> dict | None:
    """Generate answer and compute evaluation metrics."""
    try:
        answer_dict = orchestrator.answer(question)
        result_metrics = orchestrator.evaluate(
            answer=answer_dict['answer'],
            context_passages=answer_dict.get('context_passages', []),
            question=question,
            reference_answer=reference,
        )
        return {
            "question": question,
            "generated_answer": answer_dict['answer'],
            "reference_answer": reference,
            "metrics": result_metrics,
            "sources": answer_dict["sources"],
        }
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def list_documents() -> dict | None:
    """List all ingested documents."""
    try:
        docs = orchestrator.list_documents()
        return {"documents": docs, "count": len(docs)}
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def ingest_documents(urls: list[str]) -> dict | None:
    """Ingest one or more Wikipedia documents."""
    try:
        success_count, failure_count, results = orchestrator.add_documents(urls)
        return {"success_count": success_count, "failure_count": failure_count, "results": results}
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def delete_documents(doc_ids: list[str] | None = None, delete_all: bool = False) -> dict | None:
    """Delete one or more documents."""
    try:
        if delete_all:
            deleted_count, message = orchestrator.delete_all_documents()
            return {"success": deleted_count > 0, "message": message, "deleted_count": deleted_count}
        elif doc_ids:
            success_count, failure_count = orchestrator.delete_documents(doc_ids)
            message = f"Deleted {success_count} document(s)" + (f", {failure_count} failed" if failure_count > 0 else "")
            return {"success": success_count > 0, "message": message, "deleted_count": success_count}
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def get_random_articles(count: int) -> dict | None:
    """Get random Wikipedia article URLs."""
    try:
        articles = orchestrator.get_random_articles(count=count)
        return {"articles": articles, "count": len(articles)}
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def reset_to_defaults() -> dict | None:
    """Reset the collection to default articles."""
    try:
        success_count, failure_count, results = orchestrator.reset_to_default_documents()
        return {"success_count": success_count, "failure_count": failure_count, "results": results}
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def load_eval_example(question: str, reference: str) -> None:
    """Populate evaluation inputs from a preset example."""
    st.session_state["eval_q"] = question
    st.session_state["eval_ref"] = reference



# --- Sidebar ---
with st.sidebar:
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



# --- Main Tabs ---
tab_qa, tab_eval, tab_ingest = st.tabs(["💬 Ask Questions", "📊 Evaluate Answers", "🗂️ Manage Documents"])

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
                cols[0].metric("Faithfulness", f"{metrics['faithfulness']:.4f}")
                cols[1].metric("Context Precision", f"{metrics['context_precision']:.4f}")
                cols[2].metric("Answer Relevance", f"{metrics['answer_relevance']:.4f}")
                if "context_recall" in metrics:
                    cols[3].metric("Context Recall", f"{metrics['context_recall']:.4f}")
                if "answer_correctness" in metrics:
                    cols[4].metric("Answer Correctness", f"{metrics['answer_correctness']:.4f}")

                st.subheader("Retrieved Sources")
                for i, source in enumerate(result["sources"], 1):
                    with st.expander(f"[{i}] {source['title']} (score: {source['score']:.4f})"):
                        st.write(source["text"])
                        if source.get("url"):
                            st.caption(f"Source: {source['url']}")

with tab_ingest:
    st.markdown("""
    **Manage Documents** — Add Wikipedia articles to your knowledge base or remove existing ones.
    
    Only valid Wikipedia links are accepted (e.g., `https://en.wikipedia.org/wiki/Wikipedia:Article_titles`).
    """)
    
    # --- List current documents ---
    st.subheader("📚 Ingested Documents")
    doc_list = list_documents()
    
    if doc_list and doc_list["documents"]:
        documents = doc_list["documents"]
        st.info(f"Total: {doc_list['count']} document(s)")
        
        # Create a scrollable container for the document list
        with st.container(height=400, border=True):
            # Create a table view with URLs
            for i, doc in enumerate(documents):
                col1, col2, col3 = st.columns([2, 3, 0.3])
                with col1:
                    st.write(f"**#{doc['id']}** {doc['title']}")
                with col2:
                    st.caption(doc['url'])
                with col3:
                    if st.button("🗑️", key=f"del_{doc['id']}", help="Delete this document"):
                        result = delete_documents(doc_ids=[doc['id']])
                        if result and result["success"]:
                            st.success(result["message"])
                            st.rerun()
                        elif result:
                            st.error(result["message"])
        
        # Delete all button
        if st.button("🗑️ Delete All Documents", key="del_all", help="WARNING: This cannot be undone", type="secondary"):
            result = delete_documents(delete_all=True)
            if result and result["success"]:
                st.success(result["message"])
                st.rerun()
            elif result:
                st.error(result["message"])
    else:
        st.info("📭 No documents ingested yet. Add some below!")
    
    # --- Reset to defaults option ---
    col_reset_left, col_reset_right = st.columns([1, 3])
    with col_reset_left:
        if st.button("♻️ Reset to Defaults", key="reset_defaults", help="Restore the default Hungary-related articles"):
            with st.spinner("Resetting to default articles..."):
                result = reset_to_defaults()
            if result and result["success_count"] > 0:
                st.success(f"✅ Reset complete! Added {result['success_count']} default article(s)")
                st.rerun()
            elif result:
                st.error(f"❌ Reset failed: {result.get('failure_count', 0)} article(s) failed to load")
    with col_reset_right:
        st.write("")  # Spacer
    
    st.divider()
    st.subheader("➕ Add Documents")
    st.markdown("Enter Wikipedia article URLs. Example: `https://en.wikipedia.org/wiki/Hungary`")
    
    # Single URL input
    single_url = st.text_input(
        "Enter a Wikipedia URL:",
        placeholder="https://en.wikipedia.org/wiki/Example",
        key="single_url"
    )
    
    if st.button("Add Document", key="btn_add_single", type="primary"):
        if single_url.strip():
            result = ingest_documents([single_url])
            if result:
                if result["success_count"] > 0:
                    st.success(f"✅ {result['results'][0]['message']}")
                    st.rerun()
                else:
                    st.error(f"❌ {result['results'][0]['message']}")
        else:
            st.warning("Please enter a URL")
    
    st.divider()
    
    # Random articles loader
    st.subheader("🎲 Load Random Articles")
    st.markdown("Automatically add random Wikipedia articles to your collection:")
    
    random_pick_div, _ = st.columns([1, 4])
    with random_pick_div:
        new_article_count = st.number_input(
                "Number of articles:",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                key="random_count"
            )
        if st.button(f"🎲 Load", key="btn_random", type="primary"):
            with st.spinner(f"Fetching {new_article_count} random article(s)..."):
                random_result = get_random_articles(count=new_article_count)
            
            if random_result and random_result["articles"]:
                # Ingest the random articles
                with st.spinner("Adding articles..."):
                    ingest_result = ingest_documents(random_result["articles"])
                
                if ingest_result:
                    if ingest_result["success_count"] > 0:
                        st.success(f"✅ Added {ingest_result['success_count']} article(s)")
                        st.rerun()
                    if ingest_result["failure_count"] > 0:
                        st.warning(f"⚠️ Failed to add {ingest_result['failure_count']} article(s)")
            else:
                st.info("No new articles available to load.")