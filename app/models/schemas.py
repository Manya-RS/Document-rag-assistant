from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    document_name: str = Field(
        ...,
        description="Name of the uploaded document."
    )

    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask about the document."
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of relevant chunks to retrieve."
    )


class SourceItem(BaseModel):
    source: str
    chunk_id: int
    distance: float
    content: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]