## To download llm model : ##

https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q2_K.gguf

## Explanation of each file ##

- fc2.py, function_call_llama.py are base code for function calling
- the rest are base code for RAG

## How to run chatbot ##

``` uvicorn app:app --reload --host 0.0.0.0 --port 8000 ```

## Install libraries needed ##

``` pip install -r requirements.txt ```