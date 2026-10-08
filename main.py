from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os

from pdf_loader import load_pdf
from vector_store import create_vector_store
from rag_chain import ask_question

app = FastAPI()

vector_db = None


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    global vector_db

     
    os.makedirs("uploads", exist_ok=True)

    file_path = f"uploads/{file.filename}"

    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = load_pdf(file_path)

    vector_db = create_vector_store(text)

    return {"message": "PDF processed successfully"}


@app.get("/ask")
def ask(question: str):

    if vector_db is None:
        raise HTTPException(status_code=400, detail="No documents processed. Upload a PDF first.")

    answer = ask_question(vector_db, question)

    return {"answer": answer}