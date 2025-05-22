import os
from PyPDF2 import PdfReader
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import chromadb
from chromadb.config import Settings
from llama_cpp import Llama  # opsional, tergantung nanti RAG kamu

# --- STEP 1: Load PDF ---
def load_pdf_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# --- STEP 2: Split text into chunks ---
def split_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# --- STEP 3: Setup ChromaDB ---

chroma_client = chromadb.Client() 
# Gunakan sentence-transformers sebagai embedding function
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = chroma_client.get_or_create_collection(
    name="rag_pdf",
    embedding_function=embedding_function
)

# --- STEP 4: Ingest PDF into Chroma ---
def ingest_pdf_to_chroma(file_path):
    text = load_pdf_text(file_path)
    chunks = split_text(text)

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"chunk_{i}"]
        )

    print(f"Ingested {len(chunks)} chunks from {file_path}")

# --- RUN ---
if __name__ == "__main__":
    pdf_path = "pajak_hotel.pdf"  # ganti dengan path PDF kamu
    ingest_pdf_to_chroma(pdf_path)
