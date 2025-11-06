# image_knowledge_base.py
# Developer-only: builds and queries the IMAGE Knowledge Base for AROGYAM.
# Uses CLIP (SentenceTransformer) to encode and store image embeddings in ChromaDB.

import os
import uuid
from pathlib import Path
from typing import List
from PIL import Image
import chromadb
from sentence_transformers import SentenceTransformer, util

# ----------------------------------
# Configuration
# ----------------------------------
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "data" / "skin_images"        # where dataset images are placed
VECTOR_DIR = BASE_DIR / "vector_store_images"        # where embeddings will be stored
COLLECTION_NAME = "arogyam_image_kb"

# load CLIP model once
EMBED_MODEL = "clip-ViT-B-32"
_clip_model = SentenceTransformer(EMBED_MODEL)


# ----------------------------------
# Build the image KB
# ----------------------------------
def build_image_kb():
    """Encode all images in data/skin_images and store them in ChromaDB."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    image_files = [f for f in IMAGE_DIR.rglob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
    if not image_files:
        print(f"[error] No images found in: {IMAGE_DIR}")
        return

    total = 0
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert("RGB")
            emb = _clip_model.encode([img], show_progress_bar=False)
            collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=emb,
                metadatas=[{"file": img_path.name, "path": str(img_path)}]
            )
            total += 1
            print(f"[indexed] {img_path.name}")
        except Exception as e:
            print(f"[skip] {img_path.name}: {e}")

    print(f"[done] Image KB built. Total indexed images: {total}")
    print(f"[info] Stored in: {VECTOR_DIR}")


# ----------------------------------
# Search the image KB
# ----------------------------------
def search_image_kb(query_image: Image.Image, top_k: int = 5) -> List[dict]:
    """Find visually similar images from the stored image KB."""
    client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    emb_query = _clip_model.encode([query_image], show_progress_bar=False)
    res = collection.query(query_embeddings=emb_query, n_results=top_k)

    # format readable output
    out = []
    docs = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for m, d in zip(docs, dists):
        out.append({"file": m.get("file"), "path": m.get("path"), "distance": round(d, 3)})
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build or rebuild the image knowledge base.")
    parser.add_argument("--test", type=str, help="Test search with a given image file path.")
    args = parser.parse_args()

    if args.build:
        build_image_kb()
    elif args.test:
        test_img = Image.open(args.test).convert("RGB")
        results = search_image_kb(test_img, top_k=5)
        print("Top matches:")
        for r in results:
            print(r)
