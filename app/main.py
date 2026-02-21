from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import generate_answer

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.on_event("startup")
def load_models():
    from app.embeddings import get_model
    get_model()

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/query")
def query_endpoint(request: QueryRequest):
    result = generate_answer(request.question)
    return result