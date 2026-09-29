from pathlib import Path
from pypdf import PdfReader


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


def extract_text(file_path: str) -> str:
    """
    Extract text from PDF, TXT, or Markdown files.
    """

    path = Path(file_path)
    extension = path.suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Unsupported file type. Only PDF, TXT, and Markdown files are allowed."
        )

    if extension == ".pdf":
        reader = PdfReader(str(path))

        pages = []

        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)

        return "\n".join(pages).strip()

    if extension in {".txt", ".md"}:
        return path.read_text(
            encoding="utf-8",
            errors="replace"
        ).strip()

    return ""