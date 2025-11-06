# knowledge_base.py
# Developer-only KB builder and retriever for AROGYAM.

import os
import argparse
import uuid
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# ---------- Paths resolved relative to this file ----------
BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "vector_store"
DATA_DIR = BASE_DIR / "data" / "knowledge"
COLLECTION_NAME = "arogyam_kb"

# ---------- Embedding model ----------
EMBED_MODEL = "all-MiniLM-L6-v2"
_embedder = SentenceTransformer(EMBED_MODEL)

def _extract_text_from_pdf(file_path: Path) -> str:
    text_parts = []
    reader = PdfReader(str(file_path))
    for page in reader.pages:
        t = page.extract_text() or ""
        if t:
            text_parts.append(t.replace("\n", " "))
    return " ".join(text_parts)

def _read_text_file(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore").replace("\n", " ")

def _chunk_text(text: str, chunk_size: int = 400) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def build_kb() -> None:
    # Ensure directories exist
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Collect input files (pdf/txt) recursively
    files = list(DATA_DIR.rglob("*.pdf")) + list(DATA_DIR.rglob("*.txt"))
    if not files:
        print(f"[error] No PDFs or TXTs found in: {DATA_DIR}")
        print("Add files under data/knowledge/ and re-run: python knowledge_base.py --build")
        return

    client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    total_chunks = 0
    for fpath in files:
        try:
            if fpath.suffix.lower() == ".pdf":
                text = _extract_text_from_pdf(fpath)
            else:
                text = _read_text_file(fpath)

            if not text.strip():
                print(f"[skip] No extractable text: {fpath.name}")
                continue

            chunks = _chunk_text(text, chunk_size=400)
            if not chunks:
                print(f"[skip] No chunks produced: {fpath.name}")
                continue

            embeddings = _embedder.encode(chunks, show_progress_bar=False)
            ids = [str(uuid.uuid4()) for _ in chunks]

            collection.add(documents=chunks, embeddings=embeddings, ids=ids)
            total_chunks += len(chunks)
            print(f"[indexed] {fpath.name}: {len(chunks)} chunks")

        except Exception as e:
            print(f"[error] {fpath.name}: {e}")

    print(f"[done] KB built. Total chunks: {total_chunks}")
    print(f"[info] Vector store at: {VECTOR_DIR.resolve()}")

def search_kb(query: str, top_k: int = 5) -> list[str]:
    client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    q_emb = _embedder.encode([query])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=top_k)
    return res.get("documents", [[]])[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Rebuild KB from data/knowledge/")
    args = parser.parse_args()
    if args.build:
        print(f"[info] BASE_DIR: {BASE_DIR}")
        print(f"[info] DATA_DIR: {DATA_DIR}")
        print(f"[info] VECTOR_DIR: {VECTOR_DIR}")
        build_kb()
