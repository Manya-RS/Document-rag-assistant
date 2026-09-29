from typing import TypedDict
import re

from langgraph.graph import StateGraph, START, END

from app.retrieval.retriever import retrieve
from app.generation.generator import generate_answer


class RAGState(TypedDict):
    document_name: str
    question: str
    top_k: int
    retrieved_chunks: list[dict]
    answer: str
    sources: list[dict]


def retrieve_node(state: RAGState) -> dict:
    """
    Retrieve a larger candidate set for better reranking.
    """

    candidate_k = min(state["top_k"] * 2, 10)

    chunks = retrieve(
        document_name=state["document_name"],
        query=state["question"],
        top_k=candidate_k
    )

    return {
        "retrieved_chunks": chunks
    }


def rerank_node(state: RAGState) -> dict:
    """
    Rerank retrieved chunks using lexical query-term overlap
    combined with the FAISS semantic distance.
    """

    question_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]{3,}\b",
            state["question"].lower()
        )
    )

    scored_chunks = []

    for item in state["retrieved_chunks"]:

        chunk_words = set(
            re.findall(
                r"\b[a-zA-Z0-9]{3,}\b",
                item["chunk"].lower()
            )
        )

        overlap = len(question_words & chunk_words)

        scored_chunks.append({
            **item,
            "lexical_overlap": overlap
        })

    scored_chunks.sort(
        key=lambda item: (
            -item["lexical_overlap"],
            item["distance"]
        )
    )

    return {
        "retrieved_chunks": scored_chunks[:state["top_k"]]
    }


def generate_node(state: RAGState) -> dict:
    """
    Generate a grounded answer using reranked context.
    """

    answer = generate_answer(
        question=state["question"],
        context=state["retrieved_chunks"]
    )

    sources = []

    for index, result in enumerate(
        state["retrieved_chunks"]
    ):
        sources.append({
            "source": state["document_name"],
            "chunk_id": index + 1,
            "distance": result["distance"],
            "content": result["chunk"]
        })

    return {
        "answer": answer,
        "sources": sources
    }


workflow = StateGraph(RAGState)

workflow.add_node(
    "retrieve",
    retrieve_node
)

workflow.add_node(
    "rerank",
    rerank_node
)

workflow.add_node(
    "generate",
    generate_node
)

workflow.add_edge(
    START,
    "retrieve"
)

workflow.add_edge(
    "retrieve",
    "rerank"
)

workflow.add_edge(
    "rerank",
    "generate"
)

workflow.add_edge(
    "generate",
    END
)

rag_graph = workflow.compile()