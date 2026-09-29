from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import QueryRequest, QueryResponse
from app.workflow.graph import rag_graph
from app.retrieval.retriever import retrieve
from app.generation.generator import stream_answer

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse
)
async def query_document(request: QueryRequest):
    """
    Ask a question about a specific uploaded document
    using the LangGraph RAG workflow.
    """

    try:
        result = rag_graph.invoke({
            "document_name": request.document_name,
            "question": request.question,
            "top_k": request.top_k,
            "retrieved_chunks": [],
            "answer": "",
            "sources": []
        })

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process the question: {error}"
        )
@router.post("/query/stream")
def stream_query(request: QueryRequest):
    """
    Stream a grounded answer from the selected document.
    """

    try:
        context = retrieve(
            document_name=request.document_name,
            query=request.question,
            top_k=request.top_k
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve document context: {error}"
        )

    def generate():
        try:
            yield from stream_answer(
                question=request.question,
                context=context
            )
        except Exception as error:
            yield f"\n\n[Streaming error: {error}]"

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )