# main.py
import json
import os
from typing import List, Dict, Any
import numpy as np

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn

# embeddings + faiss
from sentence_transformers import SentenceTransformer
import faiss

# LLM
from llama_cpp import Llama

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "mistral-7b-instruct-v0.1.Q2_K.gguf")
KB_PATH = os.getenv("KB_PATH", "knowledge_bank.json")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
TOP_K = int(os.getenv("TOP_K", "3"))
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.65"))  # cosine similarity threshold

# ----------------------------
# Load knowledge bank
# ----------------------------
with open(KB_PATH, "r", encoding="utf-8") as f:
    KB: List[Dict[str, str]] = json.load(f)

questions = [item["question"] for item in KB]
answers = [item["answer"] for item in KB]

# ----------------------------
# Load embedding model
# ----------------------------
print("Loading embedding model:", EMBED_MODEL)
embedder = SentenceTransformer(EMBED_MODEL)

# Encode corpus (questions) -> numpy array
corpus_embeddings = embedder.encode(questions, convert_to_numpy=True)
# normalize for cosine
corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

dim = corpus_embeddings.shape[1]
# FAISS index with Inner Product (for cosine similarity after normalizing)
faiss_index = faiss.IndexFlatIP(dim)
faiss_index.add(corpus_embeddings)
print(f"FAISS index ready — {faiss_index.ntotal} vectors, dim={dim}")

# ----------------------------
# Load LLM (Mistral via llama_cpp)
# ----------------------------
print("Loading LLaMA model from:", MODEL_PATH)
llm = Llama(model_path=MODEL_PATH, n_ctx=1024)

def build_prompt(question: str, contexts: List[str]) -> str:
    """
    Build a prompt that injects retrieved contexts. Keeps instruction short and Indonesian.
    """
    ctx_text = "\n\n".join(f"- {c}" for c in contexts) if contexts else ""
    if ctx_text:
        return (
            "[INST] Kamu adalah asisten AI helpdesk yang hanya menjawab dalam Bahasa Indonesia.\n"
            "Gunakan informasi resmi berikut (jika relevan) untuk menjawab pertanyaan.\n\n"
            f"{ctx_text}\n\nPertanyaan: {question}\n\nJawab dengan singkat, jelas, dan ramah. [/INST]"
        )
    else:
        return f"[INST] Kamu adalah asisten AI helpdesk yang hanya menjawab dalam Bahasa Indonesia. {question} [/INST]"

def retrieve_topk(question: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Return list of top-k retrieved KB items with similarity scores (cosine).
    """
    q_emb = embedder.encode([question], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    D, I = faiss_index.search(q_emb, k)  # D: scores (IP after normalization = cosine sim)
    sims = D[0].tolist()
    idxs = I[0].tolist()
    results = []
    for sim, idx in zip(sims, idxs):
        if idx < 0 or idx >= len(KB):
            continue
        results.append({
            "question": KB[idx]["question"],
            "answer": KB[idx]["answer"],
            "score": float(sim)
        })
    return results

def ask_pipeline(question: str, top_k: int = TOP_K, sim_threshold: float = SIM_THRESHOLD) -> Dict[str, Any]:
    """
    Retrieval-augmented pipeline:
    - retrieve top_k
    - if top result similarity >= sim_threshold, include retrieved answers as context
    - call LLM with contexts and return answer + metadata
    """
    retrieved = retrieve_topk(question, k=top_k)
    contexts = []
    used_from_kb = False
    if retrieved:
        # Collect contexts whose score >= small floor (we'll send top_k anyway)
        contexts = [r["answer"] for r in retrieved]
        if retrieved[0]["score"] >= sim_threshold:
            used_from_kb = True

    prompt = build_prompt(question, contexts)
    # You can tune max_tokens / temperature as needed
    out = llm(prompt, max_tokens=256, temperature=0.0)
    text = out.get("choices", [{}])[0].get("text", "").strip()

    return {
        "answer": text,
        "source": "kb+llm" if contexts else "llm",
        "retrieved": retrieved,
    }

# ----------------------------
# FastAPI app + template
# ----------------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index():
    # return the static template file content
    return templates.TemplateResponse("index.html", {"request": {}})

@app.post("/ask")
async def ask_api(request: Request):
    payload = await request.json()
    question = payload.get("question", "")
    if not question:
        return JSONResponse({"error": "question empty"}, status_code=400)
    result = ask_pipeline(question)
    return JSONResponse(result)

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
