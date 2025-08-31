from PyPDF2 import PdfReader
from llama_cpp import Llama
# 1. Load model Mistral
llm = Llama(model_path="mistral-7b-instruct-v0.1.Q2_K.gguf", n_ctx=1024)


reader = PdfReader("pajak_hotel.pdf")
context = ""
for page in reader.pages:
    context += page.extract_text()
#print(text)

# 3.prompt RAG
pertanyaan = "Apa itu Pajak Jasa Perhotelan?"
prompt = f"""[INST] Berdasarkan teks berikut:

\"\"\"{context}\"\"\"

Jawablah pertanyaan ini dalam Bahasa Indonesia:
{pertanyaan}
[/INST]
"""

# 4. Generate jawaban
output = llm(prompt, max_tokens=200)
print(output["choices"][0]["text"].strip())

