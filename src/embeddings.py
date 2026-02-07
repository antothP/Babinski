import ollama
import numpy as np

def get_embeddings(chunk, vector_store):
    response = ollama.embed(model="embeddinggemma", input=chunk)
    vector_store.append(response['embeddings'])
    