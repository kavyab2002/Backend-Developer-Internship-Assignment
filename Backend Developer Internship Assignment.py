from fastapi import FastAPI, File, UploadFile
from typing import List

from sentence_transformers import SentenceTransformer, util

from PyPDF2 import PdfFileReader
import os
import sqlite3

app = FastAPI()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Implement file saving and processing
    return {"filename": file.filename}

@app.get("/docs")
async def search_documents(q: str):
    # Implement search logic
    return {"query": q, "results": []}

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PdfFileReader(file)
        text = ''
        for page_num in range(reader.numPages):
            text += reader.getPage(page_num).extract_text()
        return text

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(text):
    return model.encode(text, convert_to_tensor=True)

def search(query, documents):
    query_embedding = generate_embeddings(query)
    document_embeddings = [generate_embeddings(doc['text']) for doc in documents]
    scores = util.pytorch_cos_sim(query_embedding, document_embeddings)
    return scores


def create_database():
    conn = sqlite3.connect('documents.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE documents (id INTEGER PRIMARY KEY, name TEXT, text TEXT)''')
    conn.commit()
    conn.close()

def add_document(name, text):
    conn = sqlite3.connect('documents.db')
    c = conn.cursor()
    c.execute("INSERT INTO documents (name, text) VALUES (?, ?)", (name, text))
    conn.commit()
    conn.close()

def get_all_documents():
    conn = sqlite3.connect('documents.db')
    c = conn.cursor()
    c.execute("SELECT * FROM documents")
    documents = c.fetchall()
    conn.close()
    return documents

