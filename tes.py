import os
from PyPDF2 import PdfReader
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import chromadb
from chromadb.config import Settings
from llama_cpp import Llama  # pastikan sudah install dan model siap

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

# --- STEP 3: Setup ChromaDB client dan collection ---
# NOTE: jika error deprecated config, cek dokumentasi migrasi chroma terbaru
chroma_client = chromadb.Client()
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = chroma_client.get_or_create_collection(
    name="rag_pdf",
    embedding_function=embedding_function
)

# --- STEP 4: Ingest PDF ke ChromaDB ---
def ingest_pdf_to_chroma(file_path):
    text = load_pdf_text(file_path)
    chunks = split_text(text)

    # Hapus dulu data lama jika perlu
    # collection.delete()

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"chunk_{i}"]
        )

    print(f"Ingested {len(chunks)} chunks from {file_path}")

# --- STEP 5: Query + Generate jawaban pake Llama ---
def answer_question(question, top_k=3):
    # Query ke collection
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
    )

    # Ambil dokumen (chunk) hasil retrieval
    retrieved_chunks = results['documents'][0]  # karena 1 query
    context = "\n".join(retrieved_chunks)

    # Setup Llama (pastikan model_path sudah sesuai)
    llm = Llama(model_path="mistral-7b-instruct-v0.1.Q2_K.gguf", n_ctx=512)

    # Buat prompt untuk generate jawaban berdasarkan konteks
    prompt = f"Berikan jawaban berdasarkan konteks berikut:\n{context}\n\nPertanyaan: {question}\nJawaban:"

    # Generate jawaban
    response = llm(prompt, max_tokens=200, temperature=0.1)
    return response['choices'][0]['text'].strip()

# --- RUN ---
if __name__ == "__main__":
    pdf_path = "pajak_hotel.pdf"  # ganti path PDF kamu

    # Ingest PDF ke ChromaDB (jalanin sekali saja, kalau sudah ingest bisa skip)
    ingest_pdf_to_chroma(pdf_path)

    # Tes tanya jawab
    pertanyaan = "Apa itu pajak hotel?"
    jawaban = answer_question(pertanyaan)
    print("\n=== Jawaban ===")
    print(jawaban)
