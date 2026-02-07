import ollama
import numpy as np

def get_embeddings(chunk):
    response = ollama.embed(model="embeddinggemma", input=chunk)
    return response['embeddings']
