from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import search_docs, generate_answer
from app.logger import log_query
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app) # monitoring prometheus
class Query(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "RAG API running 🚀"}

@app.post("/ask")
def ask(query: Query):

    context = search_docs(query.question)
    answer = generate_answer(query.question, context)

    # log
    log_query(query.question, answer)

    return {
        "question": query.question,
        "answer": answer,
        "context": context[:500]
    }