import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the .env file.")

client = genai.Client(api_key=api_key)


GENERATION_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite"
]


def generate_answer(
    question: str,
    context: list[dict]
) -> str:
    """
    Generate a grounded answer using retrieved document context.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not context:
        return "I could not find relevant information in the selected document."

    context_text = "\n\n".join(
        [
            f"Context {index + 1}:\n{item['chunk']}"
            for index, item in enumerate(context)
        ]
    )

    prompt = f"""
You are a document-based AI assistant.

Answer the user's question using ONLY the provided document context.

Rules:
1. Do not use outside knowledge.
2. Do not invent or assume information.
3. If the answer is not present in the context, clearly say that the information is not available in the provided document.
4. Give a clear and concise answer.
5. When possible, explain the answer using the terminology from the document.

Document context:
{context_text}

User question:
{question}

Answer:
"""

    last_error = None

    for model in GENERATION_MODELS:
        try:
            print(f"Trying generation model: {model}")

            response = client.models.generate_content(
                model=model,
                contents=prompt
            )

            if response.text:
                return response.text.strip()

        except Exception as error:
            last_error = error
            print(
                f"Generation failed with {model}: {error}"
            )

    if last_error:
        raise last_error

    raise RuntimeError("No answer was generated.")
def stream_answer(
    question: str,
    context: list[dict]
):
    """
    Stream a grounded answer using retrieved document context.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not context:
        yield "I could not find relevant information in the selected document."
        return

    context_text = "\n\n".join(
        [
            f"Context {index + 1}:\n{item['chunk']}"
            for index, item in enumerate(context)
        ]
    )

    prompt = f"""
You are a document-based AI assistant.

Answer the user's question using ONLY the provided document context.

Rules:
1. Do not use outside knowledge.
2. Do not invent or assume information.
3. If the answer is not present in the context, clearly say that the information is not available in the provided document.
4. Give a clear and concise answer.
5. When possible, explain the answer using the terminology from the document.

Document context:
{context_text}

User question:
{question}

Answer:
"""

    last_error = None

    for model in GENERATION_MODELS:
        try:
            print(f"Trying streaming model: {model}")

            stream = client.models.generate_content_stream(
                model=model,
                contents=prompt
            )

            for chunk in stream:
                if chunk.text:
                    yield chunk.text

            return

        except Exception as error:
            last_error = error
            print(
                f"Streaming failed with {model}: {error}"
            )

    if last_error:
        raise last_error

    raise RuntimeError("No streamed answer was generated.")