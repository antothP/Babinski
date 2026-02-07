import ollama

def get_embeddings(chunk, vector_store):
    response = ollama.embeddings(model="embeddinggamma", input=chunk)
    vector_store.append(response['embedding'])
