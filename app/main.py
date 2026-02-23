from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from app.rag import generate_answer

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    from app.embeddings import get_model
    get_model()
    yield
    # Shutdown (cleanup if needed)

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/query")
def query_endpoint(request: QueryRequest):
    try:
        result = generate_answer(request.question)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))