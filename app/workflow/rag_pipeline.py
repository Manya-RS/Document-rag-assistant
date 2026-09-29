from app.retrieval.retriever import retrieve
from app.generation.generator import generate_answer


def answer_question(
    document_name: str,
    question: str,
    top_k: int = 3
) -> dict:
    """
    Retrieve relevant document chunks and generate
    a grounded answer from them.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    # 1. Retrieve relevant chunks
    retrieved_chunks = retrieve(
        document_name=document_name,
        query=question,
        top_k=top_k
    )

    # 2. Generate grounded answer
    answer = generate_answer(
        question=question,
        context=retrieved_chunks
    )

    # 3. Prepare source information
    sources = []

    for index, result in enumerate(retrieved_chunks):
        sources.append({
            "source": document_name,
            "chunk_id": index + 1,
            "distance": result["distance"],
            "content": result["chunk"]
        })

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }