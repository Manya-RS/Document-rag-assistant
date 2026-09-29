import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the .env file.")

client = genai.Client(api_key=api_key)


EMBEDDING_MODEL = "gemini-embedding-001"


def generate_embedding(
    text: str,
    task_type: str = "RETRIEVAL_DOCUMENT"
) -> list[float]:
    """Generate an embedding for text."""

    if not text or not text.strip():
        raise ValueError("Cannot generate an embedding for empty text.")

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=768
        )
    )

    return response.embeddings[0].values

def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple document chunks."""

    if not texts:
        return []

    embeddings = []

    for text in texts:
        embeddings.append(generate_embedding(text))

    return embeddings