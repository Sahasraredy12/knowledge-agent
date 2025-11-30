import os
import uuid
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

DOCS_DIR = "docs"
DB_DIR = "vectorstore"
COLLECTION_NAME = "knowledge_base"


def infer_doc_type(filename):
    name = filename.lower()
    if any(x in name for x in ["leave", "policy", "handbook", "hr"]):
        return "hr"
    if any(x in name for x in ["yoga", "fracture", "vision", "project"]):
        return "project"
    return "other"


def load_docs():
    documents = []
    for file in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            items = loader.load()
        elif file.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            items = loader.load()
        else:
            continue

        doc_type = infer_doc_type(file)

        for d in items:
            d.metadata["source"] = file
            d.metadata["doc_type"] = doc_type
            documents.append(d)

    return documents


def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    return splitter.split_documents(documents)


def main():
    print("🚀 Starting ingestion...")

    docs = load_docs()
    print(f"Loaded: {len(docs)} docs")

    chunks = split_docs(docs)
    print(f"Chunks: {len(chunks)}")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c.page_content for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)
    ids = [str(uuid.uuid4()) for _ in texts]
    metadatas = [c.metadata for c in chunks]

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

    print("🎉 Ingestion complete!")


if __name__ == "__main__":
    main()
