from pathlib import Path
import json
import faiss

from app.ingestion.document_loader import extract_text
from app.ingestion.chunker import chunk_text
from app.retrieval.embeddings import generate_embeddings


VECTORSTORE_DIR = Path("vectorstore")
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


def index_document(file_path: str):
    """
    Process a document and create a persistent FAISS index.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    # 1. Extract text
    text = extract_text(str(path))

    if not text:
        raise ValueError("No readable text found in the document.")

    # 2. Split into chunks
    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("No chunks were created from the document.")

    # 3. Generate embeddings
    embeddings = generate_embeddings(chunks)

    if not embeddings:
        raise ValueError("No embeddings were generated.")

    # 4. Convert embeddings to FAISS format
    import numpy as np

    vectors = np.array(
        embeddings,
        dtype="float32"
    )

    # 5. Create FAISS index
    dimension = len(embeddings[0])

    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    # 6. Create document-specific folder
    document_name = path.stem

    document_dir = VECTORSTORE_DIR / document_name
    document_dir.mkdir(parents=True, exist_ok=True)

    # 7. Save FAISS index
    index_path = document_dir / "index.faiss"
    faiss.write_index(index, str(index_path))

    # 8. Save chunks
    chunks_path = document_dir / "chunks.json"

    with open(
        chunks_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    return {
        "document": path.name,
        "characters": len(text),
        "chunks": len(chunks),
        "embedding_dimension": dimension,
        "index_path": str(index_path),
        "chunks_path": str(chunks_path)
    }