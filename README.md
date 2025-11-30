📚 Knowledge Base QA Agent

An intelligent question–answering agent that reads PDFs & text files, builds embeddings, stores them in a Chroma vector database, and answers questions using a local LLM (Ollama + llama3) with retrieval‑augmented generation (RAG).

---

## 🚀 Overview of the Agent

This project lets you load company documents such as HR policies, onboarding guides, FAQs, and project descriptions into a vector store and ask natural‑language questions about them.  
The agent retrieves the most relevant chunks using semantic similarity and then asks a local llama3 model (via Ollama) to generate short, clear answers grounded in those documents.

You can switch between different “workspaces” (HR docs, project docs, or all docs), see which documents were used as sources, and chat with the agent over multiple turns.

---

## ✨ Features & Limitations

### Features

- **Document ingestion**
  - Supports PDF and TXT files placed in the `docs/` folder.
  - Automatic chunking of documents with overlap for better context.

- **Semantic retrieval**
  - Uses `SentenceTransformer` (`all-MiniLM-L6-v2`) to generate embeddings.
  - Stores embeddings in a local **Chroma** vector database.
  - Retrieves top‑k most relevant chunks using cosine similarity.

- **LLM‑powered answers (RAG)**
  - Sends the user question + retrieved context to **llama3** running locally via **Ollama**.
  - Generates concise, 2–4 sentence answers grounded in the retrieved chunks.
  - Gracefully says “don’t know” when the answer is not in the documents.

- **Workspaces**
  - **All** – search across every ingested document.
  - **HR Docs** – focuses on leave policy, handbook, onboarding, and HR FAQs.
  - **Projects** – focuses on project PDFs (e.g., yoga pose detection, fracture detection).

- **Chat experience**
  - Per‑session chat history so follow‑up questions stay in context.
  - Sidebar button to **clear chat history** at any time.
  - “Sources” expander that shows the top‑matching chunks and their document names/types.

- **Simple UI**
  - Built with **Streamlit**: single‑page app with question box, workspace selector, answer area, sources panel, and chat transcript.

### Limitations

- **Local only LLM**  
  Requires Ollama and a local llama3 model running on the same machine.
- **No authentication / multi‑user separation**  
  Designed as a single‑user demo app, not a production multi‑tenant service.
- **Limited document types**  
  Currently targets PDFs and plain text files; no Word, Excel, or web URLs yet.
- **Heuristic document typing**  
  HR vs Project vs Other is inferred via simple filename/text keywords, not a robust classifier.
- **No online updates**  
  New documents require re‑running the ingestion script (or adding incremental ingestion logic).

---

## ⚙ Tech Stack & APIs Used

- **Language & Core**
  - Python 3.x

- **User Interface**
  - **Streamlit** – reactive web UI for the QA interface and chat history.

- **Embeddings & Retrieval**
  - **SentenceTransformers** – `all-MiniLM-L6-v2` model for text embeddings.
  - **Chroma** – local vector database for storing and querying embeddings.
  - **NumPy + scikit‑learn** – cosine similarity over stored embedding vectors.

- **Document Handling**
  - PDF loaders (via LangChain community integrations).
  - Recursive character text splitter for chunking.

- **LLM / RAG**
  - **Ollama** – to run **llama3** locally.
  - Python client for Ollama to send prompts and receive responses.

No paid external APIs are required; everything runs locally on your machine.

---

## 📂 Project Structure

knowledge-agent/
│── app.py # Streamlit app: UI, retrieval, Ollama-based answering, workspaces, chat
│── ingest.py # Ingests documents, builds embeddings + metadata, stores in Chroma
│── docs/ # PDF/TXT documents to ingest (HR, projects, etc.)
│── vectorstore/ # Auto-created Chroma DB (can be deleted and regenerated)
│── requirements.txt # Python dependencies
│── architecture.png # High-level architecture diagram of the agent
│── README.md # Project documentation (this file)
│── .gitignore # Ignore vectorstore, venv, etc.

text

---

## 🛠 Setup & Run Instructions

Follow these steps to run the project on your local machine.

### 1. Clone the repository

git clone https://github.com/<your-username>/knowledge-agent.git
cd knowledge-agent

text

### 2. Create and activate a virtual environment (recommended)

python -m venv .venv

text

**Windows:**

.venv\Scripts\activate


**macOS / Linux:**

source .venv/bin/activate

### 3. Install dependencies

pip install -r requirements.txt


### 4. Install and prepare Ollama + llama3

1. Install Ollama from the official website and complete the setup.  
2. Make sure Ollama is running (it usually starts a local service).  
3. Download the llama3 model:

ollama pull llama3

### 5. Add documents

Place your company or sample documents into the `docs/` folder, for example:

- `hr-faq.txt`
- `leave-policy.pdf`
- `employee-handbook.pdf`
- `onboarding-guide.pdf`
- `yoga_pose_detection_overview.pdf`
- `fracture_detection_project.pdf`

### 6. Run the ingestion script

This parses documents, chunks them, creates embeddings, and stores them in Chroma.

python ingest.py

If you add or change documents later, run this command again.  
You can delete the `vectorstore/` folder to force a full rebuild if needed.

### 7. Start the Streamlit app

streamlit run app.py

Open the URL shown in the terminal (usually `http://localhost:8501`).

### 8. Use the agent

1. Select a **Workspace** from the sidebar:
   - `All`, `HR Docs`, or `Projects`.
2. Type your question (for example, “What is the notice period?” or “What does the yoga project do?”).
3. Click **Get Answer**.
4. Explore:
   - The concise answer generated by the agent.
   - **Sources (top matches)** to see which documents were used.
   - **Chat history** to follow the conversation.
5. Click **🧹 Clear chat history** in the sidebar to reset the conversation and start fresh.

## 🚀 Potential Improvements

- **Richer RAG & ranking**
  - Add re‑ranking of retrieved chunks using another model.
  - Use similarity + keyword filters or hybrid search (BM25 + vectors).

- **Better workspace & metadata handling**
  - Store and display more metadata (author, date, version).
  - Allow dynamic tagging and filtering (e.g., “Policy”, “Project”, “Onboarding”).

- **Advanced chat features**
  - Multi‑session chat with saved histories.
  - User identities and per‑user context.

- **Document ingestion enhancements**
  - Upload files directly from the Streamlit UI.
  - Support for more formats: DOCX, HTML, CSV, etc.
  - Background ingestion jobs for large document sets.

- **Deployment**
  - Containerize with Docker.
  - Deploy on a small cloud VM with Ollama + Streamlit.
  - Add simple authentication for internal company use.

- **Evaluation & logging**
  - Log questions, answers, and sources for offline evaluation.
  - Add basic metrics: hit rate, average latency, and feedback thumbs‑up/down.

---

This project demonstrates a complete, end‑to‑end RAG pipeline using only local components (Chroma + Ollama), wrapped in a clean Streamlit interface—ideal as a portfolio‑ready AI agent for HR, operations, or internal knowledge bases.