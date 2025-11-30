📚 Knowledge Base QA Agent (Groq)
=================================

An intelligent question-answering agent that reads PDFs & text files, builds embeddings, stores them in a Chroma vector database, and answers questions using **Groq** LLM (llama-3.1-8b-instant) with retrieval-augmented generation (RAG).

---

## 🚀 Overview of the Agent

This project allows you to load internal company documents such as HR policies, onboarding guides, FAQs, and project descriptions into a **vector store** and ask natural-language questions about them.

The agent retrieves the most relevant document chunks using semantic similarity and sends the context + user query to the **Groq LLM** (local llama-3 model) to generate concise answers.

You can select different “workspaces” (HR Docs, Project Docs, or All), see the sources used for the answers, and maintain chat history for multi-turn interactions.

---

## ✨ Features & Limitations

### Features

- **Document ingestion**
  - Supports PDF and TXT files in the `docs/` folder.
  - Automatic chunking with overlap to retain context.

- **Semantic retrieval**
  - Embeddings generated via `SentenceTransformer` (`all-MiniLM-L6-v2`).
  - Stored locally in **Chroma** vector database.
  - Retrieves top-k relevant chunks using cosine similarity.

- **LLM-powered answers (RAG)**
  - Uses **Groq** LLM (`llama-3.1-8b-instant`) for answer generation.
  - Concise, 2–4 sentence responses grounded in retrieved chunks.
  - Returns “I don’t know” if answer is missing from documents.

- **Workspaces**
  - **All** – search across every ingested document.
  - **HR Docs** – focuses on HR-related documents.
  - **Projects** – focuses on project-specific PDFs.

- **Chat experience**
  - Session-based chat history with last 5 turns retained.
  - Clear chat history button in sidebar.
  - Sources panel shows top-matching document chunks.

- **Simple UI**
  - Single-page **Streamlit** app with question box, workspace selector, answer area, sources, and chat transcript.

### Limitations

- **Local-only LLM** – Requires Groq API key and a local llama-3 model.
- **No authentication or multi-user separation** – single-user demo app.
- **Limited document types** – only PDFs and TXT files supported.
- **Heuristic document typing** – infers HR vs Project vs Other using simple keywords.
- **No automatic updates** – new documents require re-running `ingest.py`.

---

## ⚙ Tech Stack & APIs Used

- **Language & Core**
  - Python 3.x

- **User Interface**
  - **Streamlit** – reactive web UI for QA interface and chat history.

- **Embeddings & Retrieval**
  - **SentenceTransformers** – `all-MiniLM-L6-v2` for embedding generation.
  - **Chroma** – local vector database for storing and querying embeddings.
  - **NumPy + scikit-learn** – cosine similarity over embeddings.

- **Document Handling**
  - PDF and TXT loaders via LangChain community integrations.
  - Recursive character text splitter for chunking.

- **LLM / RAG**
  - **Groq Python Client** – sends prompts to local llama-3 model for responses.

- **Environment Variables**
  - `.env` file for storing **GROQ_API_KEY** securely.

---

## 📂 Project Structure

knowledge-agent/
│── app.py # Streamlit app: UI, retrieval, Groq LLM, chat
│── ingest.py # Document ingestion, chunking, embedding storage
│── docs/ # PDFs/TXT files to ingest
│── vectorstore/ # Auto-generated Chroma DB (can be deleted)
│── .env # GROQ_API_KEY (not committed)
│── requirements.txt # Python dependencies
│── .gitignore # Files to ignore in Git
│── README.md # Project documentation

---
🛠  Setup & Run Instructions

### 1. Clone the repository

git clone https://github.com/<your-username>/knowledge-agent.git
cd knowledge-agent
### 2. Create and activate virtual environment
python -m venv .venv

Windows:
.venv\Scripts\activate

macOS/Linux:
source .venv/bin/activate

### 3. Install dependencies

pip install -r requirements.txt
### 4. Configure Groq API Key
Create a .env file in the project root.

Add your Groq API key:
GROQ_API_KEY=your_api_key_here
### 5. Add documents
Place your HR or project PDFs/TXT files in the docs/ folder.

### 6. Run ingestion

python ingest.py
If you want to rebuild embeddings, delete the vectorstore/ folder first:
rm -r vectorstore
### 7. Start the Streamlit app

streamlit run app.py
Open http://localhost:xxxx in your browser.

### 8. Use the agent
Select a Workspace from the sidebar (All, HR Docs, Projects).

Ask questions in the text input box.

Click Get Answer.

Check sources in the Sources expander.

Follow conversation using Chat History.

Clear session with 🧹 Clear chat history.

---

## 🚀 Potential Improvements

Richer retrieval and ranking  
- Re-rank retrieved chunks using an additional model.  
- Combine vector search with keyword-based hybrid search for better accuracy on policy-style documents.

Enhanced workspaces and metadata  
- Store richer metadata such as author, version, and last-updated date for each document.  
- Add dynamic tagging for document types (e.g., “Policy”, “FAQ”, “Project Spec”) and allow filtering by tag.

Advanced chat  
- Support multi-session chat with saved histories that can be revisited later.  
- Add user-specific contexts and identities so the agent can personalize answers.

Document ingestion  
- Allow users to upload files directly through the Streamlit interface instead of copying into the `docs/` folder.  
- Support additional formats such as DOCX, HTML, and CSV.  
- Add background ingestion jobs for large datasets so the UI stays responsive.

Deployment  
- Containerize the app with Docker for easy deployment.  
- Deploy on a small cloud VM together with the model backend and Streamlit.  
- Add basic authentication so only internal users can access the agent.

Monitoring and evaluation  
- Log questions, answers, and retrieved sources for offline evaluation and debugging.  
- Track metrics like response quality, retrieval accuracy, and latency over time.

This project already provides a complete local RAG pipeline using a vector database plus a local LLM, wrapped in a clean Streamlit interface, making it a strong portfolio-ready AI agent for internal knowledge bases.
