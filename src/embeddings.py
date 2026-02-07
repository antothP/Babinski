import ollama
import numpy as np

def get_embeddings(chunks: list[str]) -> list[list[float]]:
    response = ollama.embed(model="embeddinggemma", input=chunks)
    return response['embeddings']
