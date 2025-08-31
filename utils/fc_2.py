from llama_cpp import Llama
import json
import re

# Load model LLaMA
llm = Llama(
    model_path="mistral-7b-instruct-v0.1.Q2_K.gguf",
    n_ctx=512,
    n_threads=4,
    verbose=False
)

# Fungsi Matematika
def tambah(a, b): return a + b
def kurang(a, b): return a - b
def kali(a, b): return a * b
def bagi(a, b): return "Tidak bisa dibagi nol" if b == 0 else a / b

# Mapping operasi
fungsi_ops = {
    "tambah": tambah,
    "kurang": kurang,
    "kali": kali,
    "bagi": bagi
}

# Prompt untuk LLaMA mengenali perintah
def get_math_json_from_prompt(perintah):
    prompt = f"""
Kamu adalah asisten yang mengekstrak operasi matematika dari kalimat.
Ambil operasi (tambah, kurang, kali, bagi) dan dua angka dari kalimat berikut.

Contoh output:
{{"operation": "kali", "a": 12, "b": 3}}

Kalimat: "{perintah}"
Output JSON:
"""
    result = llm(prompt, max_tokens=100, stop=["\n\n"])
    try:
        match = re.search(r'\{.*\}', result['choices'][0]['text'])
        if match:
            return json.loads(match.group())
        else:
            return None
    except:
        return None

# Main
def main():
    user_input = input("Masukkan perintah: ")  # contoh: "Hitung 10 dibagi 2"
    parsed = get_math_json_from_prompt(user_input)

    if parsed:
        operasi = parsed["operation"]
        a = parsed["a"]
        b = parsed["b"]
        func = fungsi_ops.get(operasi)

        if func:
            hasil = func(a, b)
            print(f"Hasil dari {operasi} {a} dan {b} adalah: {hasil}")
        else:
            print("Operasi tidak dikenali.")
    else:
        print("Gagal memahami perintah.")

if __name__ == "__main__":
    main()
