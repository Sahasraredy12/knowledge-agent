import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from sklearn.metrics.pairwise import cosine_similarity
import ollama

DB_DIR = "vectorstore"
COLLECTION_NAME = "knowledge_base"


@st.cache_resource
def load_db_and_model():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIR,
        embedding_function=None
    )
    return vectorstore, embedder


st.title("📚 Knowledge Base QA System")

vectorstore, embedder = load_db_and_model()

# 1) Workspaces
WORKSPACES = {
    "All": ["hr", "project", "other"],
    "HR Docs": ["hr"],
    "Projects": ["project"],
}

workspace = st.sidebar.selectbox("Workspace", list(WORKSPACES.keys()))

# 2) Chat history state
if "messages" not in st.session_state:
    st.session_state.messages = []  # each: {"role": "user"/"assistant", "content": str}

# Button to clear history
if st.sidebar.button("🧹 Clear chat history"):
    st.session_state.messages = []

user_question = st.text_input("Ask a question:")

# Helper: classify document type from content (simple heuristic)
def classify_doc_type(text):
    t = text.lower()
    if "leave" in t or "notice period" in t or "hrms" in t or "health insurance" in t:
        return "hr"
    if "yoga" in t or "pose detection" in t or "vision transformers" in t or "fracture" in t:
        return "project"
    return "other"


def retrieve_context(query, active_types, k=4, threshold=0.35):
    data = vectorstore._collection.get(include=["documents", "metadatas", "embeddings"])
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])
    embeds_raw = data.get("embeddings", [])

    if embeds_raw is None or len(embeds_raw) == 0:
        return None, None, "❌ No stored embeddings found. Run ingest.py again."

    # manually label type if not stored
    if not metas or len(metas) != len(docs):
        metas = []
        for d in docs:
            metas.append({
                "doc_type": classify_doc_type(d),
                "source": "unknown"
            })

    stored_embeddings = np.vstack([np.array(e) for e in embeds_raw])
    query_embedding = embedder.encode([query])
    similarities = cosine_similarity(query_embedding, stored_embeddings)[0]

    # filter by workspace type
    filtered_indices = [
        i for i, m in enumerate(metas)
        if m.get("doc_type", "other") in active_types
    ]
    if not filtered_indices:
        return None, None, "No documents for this workspace."

    filtered_sims = np.array([similarities[i] for i in filtered_indices])
    k = min(k, len(filtered_sims))
    top_local = np.argsort(filtered_sims)[-k:][::-1]
    top_indices = [filtered_indices[i] for i in top_local]

    if similarities[top_indices[0]] < threshold:
        return None, None, "No relevant information found."

    top_chunks = []
    for i in top_indices:
        meta = metas[i] if i < len(metas) else {}
        top_chunks.append({
            "text": docs[i],
            "source": meta.get("source", "unknown"),
            "doc_type": meta.get("doc_type", "other"),
        })

    context = "\n\n".join(c["text"].strip() for c in top_chunks)
    return context, top_chunks, None


def answer_with_ollama(question, context, chat_history, model_name="llama3"):
    history_text = ""
    for turn in chat_history[-5:]:  # last 5 turns
        role = turn["role"].capitalize()
        history_text += f"{role}: {turn['content']}\n"

    prompt = f"""
You are an assistant for company XYZ's internal knowledge base.

Use ONLY the information in the CONTEXT below to answer the USER's QUESTION.
If the answer is not in the context, say you don't know.

CHAT HISTORY:
{history_text}

CONTEXT:
{context}

USER QUESTION: {question}

Answer in 2–4 clear sentences.
"""
    result = ollama.generate(model=model_name, prompt=prompt)
    return result["response"]


if st.button("Get Answer") and user_question:
    # retrieve context based on workspace
    active_types = WORKSPACES[workspace]
    context, chunks, error = retrieve_context(user_question, active_types)

    if error:
        st.error(error)
    else:
        with st.spinner("Thinking with llama3..."):
            final_answer = answer_with_ollama(
                user_question,
                context,
                st.session_state.messages,
                model_name="llama3",
            )

        # update chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.session_state.messages.append({"role": "assistant", "content": final_answer})

        st.subheader("🔎 Answer")
        st.markdown(final_answer)

        # show sources with doc-type labels
        with st.expander("Sources (top matches)"):
            for c in chunks:
                label = c["doc_type"].upper()
                src = c["source"]
                preview = c["text"][:300].replace("\n", " ")
                st.markdown(f"- **[{label}]** ({src}) {preview}...")


# chat history display
if st.session_state.messages:
    st.markdown("## Chat history")
    for m in st.session_state.messages:
        prefix = "👤" if m["role"] == "user" else "🤖"
        st.markdown(f"{prefix} **{m['role'].capitalize()}:** {m['content']}")
