from llama_cpp import Llama

# --- Setup model LLaMA ---
llm = Llama(
    model_path="mistral-7b-instruct-v0.1.Q2_K.gguf",  # ganti path kalau perlu
    n_ctx=512,
    n_threads=4,
    verbose=False
)

# --- Fungsi Matematika ---
def tambah(a, b):
    return a + b

def kurang(a, b):
    return a - b

def kali(a, b):
    return a * b

def bagi(a, b):
    if b == 0:
        return "Tidak bisa dibagi nol"
    return a / b

# --- Function calling + LLaMA ---
def main():
    angka1 = 8
    angka2 = 2
    operasi = "bagi"  # ganti aja sesuai keinginan

    # Eksekusi berdasarkan pilihan
    if operasi == "tambah":
        hasil = tambah(angka1, angka2)
    elif operasi == "kurang":
        hasil = kurang(angka1, angka2)
    elif operasi == "kali":
        hasil = kali(angka1, angka2)
    elif operasi == "bagi":
        hasil = bagi(angka1, angka2)
    else:
        hasil = "Operasi tidak dikenali"

    # Buat prompt untuk LLaMA
    prompt = f"Hasil dari operasi {operasi} antara {angka1} dan {angka2} adalah {hasil}. Jelaskan hasilnya secara sederhana."

    # Kirim ke model
    response = llm(prompt, max_tokens=100, stop=["</s>"])
    
    # Tampilkan hasil
    print("=== Hasil Matematika ===")
    print(f"{angka1} {operasi} {angka2} = {hasil}")
    print("\n=== Tanggapan LLaMA ===")
    print(response["choices"][0]["text"].strip())

# Panggil main function
if __name__ == "__main__":
    main()
