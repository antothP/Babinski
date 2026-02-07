import os
from pathlib import Path
from PyPDF2 import PdfReader
from chunker import chunkeriser_texte
from embeddings import get_embeddings
from stockage import stocker_chunk, recherche_semantique, recuperer_tous_les_vecteurs, creer_schema
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from clustering import clustering
import ollama

def lire_pdf(fichier_path):
    texte = ""
    reader = PdfReader(fichier_path)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            texte += page_text + "\n"
    return texte

def main():
    dossier_data = "data"
    pdfs = list(Path(dossier_data).glob("*.pdf"))
    tous_les_chunks = []
    for pdf_path in pdfs:
        texte = lire_pdf(pdf_path)
        chunks = chunkeriser_texte(texte, chunk_size=500, overlap=50)
        for chunk in chunks:
            vector = get_embeddings(chunk)
            # tous_les_chunks.append(vector)
            stocker_chunk(chunk, {"source": str(pdf_path)}, vector)
    return tous_les_chunks

def generate_cluster_names(final_clusters, model_name="llama2"):
    cluster_names = []

    for cluster_id, chunk in final_clusters.items():
        concatenated_text = "".join([c["text"] for c in chunk if c["text"]])
        print("--- Concatenated Text ---")
        print(concatenated_text)
        print("--- End of Concatenated Text ---")
        prompt = f"""
        Voici plusieurs phrases ou extraits liés entre eux :

        {concatenated_text}

        Donne-moi un nom court et précis qui résume le thème commun de ces phrases avec le theme suivant :  IA de Confiance – Offre IA Orange Business
        Réponse :
        """
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        cluster_name = response.get("content", "").strip()
        cluster_names.append(cluster_name)
        print(f"Cluster {cluster_id} : {cluster_name}")

# chunks = main()
# print(chunks)
new_chunk = recuperer_tous_les_vecteurs()
cluster = clustering(new_chunk)
print("test")
# cluster_name = generate_cluster_names(cluster)
# print(cluster_name)



