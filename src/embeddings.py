import ollama
import numpy as np

def get_embeddings(chunk, vector_store):
    response = ollama.embed(model="embeddinggemma", input=chunk)
    vector_store.append(response['embeddings'])

def main():
    vector_store = []
    chunk = "This is a sample text chunk to be embedded."
    get_embeddings(chunk, vector_store)
    array_numpy = np.array(vector_store, dtype=np.float32)
    print(array_numpy)

main()