from fastapi import FastAPI
from app.api.upload import router as upload_router
from app.api.query import router as query_router
app = FastAPI(
    title="ShadowFox RAG Assistant",
    description="Production-style Retrieval-Augmented Generation assistant",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "ShadowFox RAG Assistant is running!"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


app.include_router(
    upload_router,
    prefix="/api"
)
app.include_router(
    query_router,
    prefix="/api"
)