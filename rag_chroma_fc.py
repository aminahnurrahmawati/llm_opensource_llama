import os
from PyPDF2 import PdfReader
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import chromadb
from chromadb.config import Settings
from llama_cpp import Llama

# STEP 1: Load PDF
def load_pdf_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# STEP 2: Split text
def split_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# STEP 3: Setup ChromaDB
chroma_client = chromadb.Client()
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = chroma_client.get_or_create_collection(
    name="rag_pdf",
    embedding_function=embedding_function
)

# STEP 4: Ingest PDF
def ingest_pdf_to_chroma(file_path):
    text = load_pdf_text(file_path)
    chunks = split_text(text)

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"doc_{i}"]
        )
    print("Ingested PDF into ChromaDB.")

# STEP 5: Query LLaMA with RAG
def ask_llama(question, top_k=3):
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    retrieved_chunks = results["documents"][0]
    context = "\n".join(retrieved_chunks)

    prompt = f"""Gunakan konteks berikut untuk menjawab pertanyaan:

{context}

Pertanyaan: {question}
Jawaban:"""

    llama = Llama(model_path="./llama-model.gguf", n_ctx=2048)
    response = llama(prompt=prompt, max_tokens=200)
    print(response["choices"][0]["text"])

# === FUNCTION CALLING STARTS HERE ===
if __name__ == "__main__":
    # Step 1: Ingest PDF
    ingest_pdf_to_chroma("contoh.pdf")

    # Step 2: Ask LLaMA
    ask_llama("Apa isi utama dari dokumen ini?")
