from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import json

from app.ingestion.document_loader import extract_text
from app.ingestion.chunker import chunk_text
from app.retrieval.indexer import index_document


router = APIRouter()

UPLOAD_DIR = Path("data/documents")
VECTORSTORE_DIR = Path("vectorstore")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT, and Markdown files are allowed."
        )

    file_path = UPLOAD_DIR / Path(file.filename).name

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    file_path.write_bytes(content)

    # --------------------------------------------------
    # Check existing index BEFORE processing the document
    # --------------------------------------------------

    document_name = file_path.stem
    document_dir = VECTORSTORE_DIR / document_name

    index_path = document_dir / "index.faiss"
    chunks_path = document_dir / "chunks.json"

    if index_path.exists() and chunks_path.exists():

        try:
            with open(
                chunks_path,
                "r",
                encoding="utf-8"
            ) as file_data:
                stored_chunks = json.load(file_data)

        except Exception:
            stored_chunks = []

        return {
            "message": "Document already indexed. Existing index reused.",
            "filename": file.filename,
            "file_type": extension,
            "characters_extracted": 0,
            "chunks_created": len(stored_chunks),
            "embedding_dimension": 768,
            "index_path": str(index_path)
        }

    # --------------------------------------------------
    # Process a new document
    # --------------------------------------------------

    try:
        extracted_text = extract_text(str(file_path))

        if not extracted_text:
            raise HTTPException(
                status_code=400,
                detail="No readable text was found in the document."
            )

        chunks = chunk_text(extracted_text)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No chunks were created from the document."
            )

        index_result = index_document(str(file_path))

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {error}"
        )

    return {
        "message": "Document uploaded, processed, and indexed successfully.",
        "filename": file.filename,
        "file_type": extension,
        "characters_extracted": len(extracted_text),
        "chunks_created": len(chunks),
        "embedding_dimension": index_result["embedding_dimension"],
        "index_path": index_result["index_path"]
    }
    