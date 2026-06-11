import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# File paths
INPUT_FILE  = "data/processed/bci_entities.json"
OUTPUT_EMBEDDINGS = "data/processed/bci_embeddings.npy"
OUTPUT_METADATA   = "data/processed/bci_metadata.json"

# Model — small, fast, runs on your GPU
MODEL_NAME = "all-MiniLM-L6-v2"

def load_papers(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        papers = json.load(f)
    print(f"Loaded {len(papers)} papers")
    return papers

def build_text(paper):
    """Combine title + abstract into one string for embedding."""
    title    = paper.get("title", "") or ""
    abstract = paper.get("abstract", "") or ""
    return f"{title}. {abstract}".strip()

def generate_embeddings(papers, model):
    print(f"\nGenerating embeddings using: {MODEL_NAME}")
    print("This may take 1-2 minutes on first run (downloading model)...")

    texts = [build_text(p) for p in papers]

    # Generate all embeddings in one batch
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        device="cuda" if __import__('torch').cuda.is_available() else "cpu"
    )

    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Each paper is now a vector of {embeddings.shape[1]} numbers")
    return embeddings

def save_outputs(papers, embeddings):
    # Save embeddings as numpy array
    np.save(OUTPUT_EMBEDDINGS, embeddings)
    print(f"Saved embeddings to {OUTPUT_EMBEDDINGS}")

    # Save metadata (title, year, citations, keywords) separately
    metadata = []
    for i, paper in enumerate(papers):
        metadata.append({
            "id"          : i,
            "title"       : paper.get("title", ""),
            "year"        : paper.get("year"),
            "citations"   : paper.get("citations", 0),
            "bci_keywords": paper.get("bci_keywords", []),
            "keyword_count": paper.get("keyword_count", 0),
            "abstract"    : paper.get("abstract", "")
        })

    with open(OUTPUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved metadata to {OUTPUT_METADATA}")

def print_summary(papers, embeddings):
    print("\n--- Embedding Summary ---")
    print(f"Total papers embedded : {len(papers)}")
    print(f"Embedding dimensions  : {embeddings.shape[1]}")
    print(f"Embedding matrix size : {embeddings.shape}")

    # Show similarity between first two papers as a sanity check
    from numpy.linalg import norm
    v1 = embeddings[0]
    v2 = embeddings[1]
    similarity = np.dot(v1, v2) / (norm(v1) * norm(v2))
    print(f"\nSanity check — similarity between paper 1 and paper 2:")
    print(f"  Paper 1: {papers[0].get('title', '')[:60]}...")
    print(f"  Paper 2: {papers[1].get('title', '')[:60]}...")
    print(f"  Cosine similarity: {similarity:.4f}  (1.0 = identical, 0.0 = unrelated)")

def main():
    print("Starting embedding generation...\n")

    # Load model
    model = SentenceTransformer(MODEL_NAME)

    # Load papers
    papers = load_papers(INPUT_FILE)

    # Generate embeddings
    embeddings = generate_embeddings(papers, model)

    # Save everything
    os.makedirs("data/processed", exist_ok=True)
    save_outputs(papers, embeddings)

    # Summary
    print_summary(papers, embeddings)

    print("\nEmbedding generation complete.")

if __name__ == "__main__":
    main()