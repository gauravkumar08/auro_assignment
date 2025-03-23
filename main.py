from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from models import ingest_document, answer_question

app = FastAPI()

class Document(BaseModel):
    content: str

class Question(BaseModel):
    question: str
    documents: List[str]

@app.post("/ingest_document")
def ingest_document_endpoint(document: Document):
    ingest_document(document.content)
    return {"message": "Document ingested successfully"}

@app.post("/answer_question")
def answer_question_endpoint(question: Question):
    answer = answer_question(question.question, question.documents)
    return {"answer": answer}