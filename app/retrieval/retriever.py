from pathlib import Path
import json
import faiss

from app.retrieval.embeddings import generate_embedding


VECTORSTORE_DIR = Path("vectorstore")


def retrieve(
    document_name: str,
    query: str,
    top_k: int = 3
):
    """
    Retrieve the most relevant chunks from a specific document.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    document_dir = VECTORSTORE_DIR / Path(document_name).stem

    index_path = document_dir / "index.faiss"
    chunks_path = document_dir / "chunks.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found for document: {document_name}"
        )

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunk data not found for document: {document_name}"
        )

    # Load FAISS index
    index = faiss.read_index(str(index_path))

    # Load chunks
    with open(
        chunks_path,
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    # Embed the user's question
    query_embedding = generate_embedding(
    query,
    task_type="RETRIEVAL_QUERY"
)

    # Search FAISS
    query_vector = [
        query_embedding
    ]

    import numpy as np

    query_vector = np.array(
        query_vector,
        dtype="float32"
    )

    distances, indices = index.search(
        query_vector,
        min(top_k, index.ntotal)
    )

    results = []

    for distance, index_position in zip(
        distances[0],
        indices[0]
    ):
        if index_position == -1:
            continue

        results.append({
            "chunk": chunks[index_position],
            "distance": float(distance)
        })

    return results