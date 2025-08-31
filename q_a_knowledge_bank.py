import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# Load embedding model untuk cari pertanyaan mirip
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load LLM
llm = Llama(model_path="mistral-7b-instruct-v0.1.Q2_K.gguf", n_ctx=512)

def prompt_indo(question, context=None):
    if context:
        return f"[INST] Kamu adalah asisten AI yang menjawab hanya dalam Bahasa Indonesia.\nGunakan informasi berikut:\n{context}\n\nPertanyaan: {question} [/INST]"
    else:
        return f"[INST] Kamu adalah asisten AI yang hanya menjawab dalam Bahasa Indonesia. {question} [/INST]"

# Load knowledge bank
with open("knowledge_bank.json", "r", encoding="utf-8") as f:
    kb = json.load(f)   # format [{"question":..., "answer":...}, ...]

# Buat index FAISS
questions = [item["question"] for item in kb]
answers = [item["answer"] for item in kb]
embeddings = embedder.encode(questions, convert_to_numpy=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def ask(question, threshold=0.5):
    q_emb = embedder.encode([question], convert_to_numpy=True)
    D, I = index.search(q_emb, 1)  # ambil 1 paling mirip
    
    if D[0][0] < threshold:  
        # kalau mirip (semakin kecil jarak, semakin dekat)
        kb_answer = answers[I[0][0]]
        return f"(Knowledge Bank) {kb_answer}"
    else:
        # kalau ga ketemu, tanya LLM
        output = llm(prompt_indo(question), max_tokens=150)
        return output["choices"][0]["text"].strip()

# --- Contoh pemakaian ---
print(ask("Apa bedanya LLM sama AI biasa?"))
print(ask("Dimana aku bisa cari informasi tambahan?"))
