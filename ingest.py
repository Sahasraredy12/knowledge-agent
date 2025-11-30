import os
import uuid
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

DOCS_DIR = "docs"
DB_DIR = "vectorstore"
COLLECTION_NAME = "knowledge_base"


def infer_doc_type(filename: str) -> str:
    """Very simple heuristic to label docs as hr / project / other."""
    name = filename.lower()
    if any(key in name for key in ["leave", "policy", "handbook", "onboarding", "hr-faq", "hr"]):
        return "hr"
    if any(key in name for key in ["yoga", "fracture", "vision", "project"]):
        return "project"
    return "other"


def load_documents():
    documents = []
    for filename in os.listdir(DOCS_DIR):
        filepath = os.path.join(DOCS_DIR, filename)
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            docs = loader.load()
        elif filename.lower().endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
        else:
            continue

        doc_type = infer_doc_type(filename)
        for d in docs:
            # store filename + type as metadata
            if d.metadata is None:
                d.metadata = {}
            d.metadata["source"] = filename
            d.metadata["doc_type"] = doc_type
            documents.append(d)

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    return chunks


def main():
    print("🚀 Starting ingestion process...")

    docs = load_documents()
    print(f"📚 Loaded documents: {len(docs)}")

    chunks = split_documents(docs)
    print(f"🔹 Created chunks: {len(chunks)}")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)
    ids = [str(uuid.uuid4()) for _ in texts]
    metadatas = [chunk.metadata for chunk in chunks]

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=None
    )

    vectorstore._collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    print("🎉 Ingestion complete! Knowledge base is ready.")


if __name__ == "__main__":
    main()
