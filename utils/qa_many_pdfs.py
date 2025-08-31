import os
from PyPDF2 import PdfReader
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import chromadb
from llama_cpp import Llama
from uuid import uuid4

# --- STEP 1: Load PDF text ---
def load_pdf_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# --- STEP 2: Split text into chunks ---
def split_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# --- STEP 3: Setup ChromaDB ---
chroma_client = chromadb.Client()
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = chroma_client.get_or_create_collection(
    name="rag_pdf_collection",
    embedding_function=embedding_function
)

# --- STEP 4: Ingest Multiple PDFs ---
def ingest_pdfs_from_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"Ingesting: {filename}")
            text = load_pdf_text(file_path)
            chunks = split_text(text)

            ids = [str(uuid4()) for _ in chunks]
            collection.add(documents=chunks, ids=ids)

# --- STEP 5: Answering Question ---
def answer_question(question, top_k=3):
    # Step 1: Search related context
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    contexts = results["documents"][0]
    context_text = "\n".join(contexts)

    # Step 2: Combine with prompt
    prompt = f"""### Instruction:
Gunakan konteks berikut untuk menjawab pertanyaan di bawah ini.

### Context:
{context_text}

### Question:
{question}

### Answer:"""

    # Step 3: Load LLM model (pastikan file .gguf valid)
    llm = Llama(
        model_path="models/llama-2-7b-chat.gguf",
        n_ctx=2048,
        n_threads=8,
        verbose=False
    )

    output = llm(prompt, max_tokens=512, stop=["###"])[
        "choices"][0]["text"].strip()
    return output


# --- MAIN PROGRAM ---
if __name__ == "__main__":
    folder_pdf = "pdfs"  # Folder isi banyak PDF
    ingest_pdfs_from_folder(folder_pdf)

    pertanyaan = input("Masukkan pertanyaan: ")
    jawaban = answer_question(pertanyaan)
    print("\nJawaban:\n", jawaban)
