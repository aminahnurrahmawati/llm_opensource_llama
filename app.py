from llama_cpp import Llama

# Load model
llm = Llama(model_path="mistral-7b-instruct-v0.1.Q2_K.gguf", n_ctx=512)

# Generate text
# prompt = "[INST] Apa itu Large Language Model? [/INST]"
# output = llm(prompt, max_tokens=100, stop=["\n"], echo=False)

def prompt_indo(question):
    return f"[INST] Kamu adalah asisten AI yang hanya menjawab dalam Bahasa Indonesia. {question} [/INST]"

output = llm(prompt_indo("Apa itu Large Language Model?"), max_tokens=100)


#print(output["choices"][0]["text"])
print(output)
