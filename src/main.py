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


def generate_cluster_names(final_clusters, model_name="gemma3"):
    cluster_names = []

    for cluster_id, chunks in final_clusters.items():
        texts = [c.get("text", "").strip() for c in chunks if c.get("text")]
        concatenated_text = " ".join(texts)
        if not concatenated_text:
            cluster_names.append("Cluster sans contenu")
            continue

        prompt = f"""
        Résume le thème commun de ces extraits en UN MOT clair.

        Extraits :
        {concatenated_text}

        Contexte : IA de Confiance – Offre IA Orange Business

        Donne uniquement le nom du thème, sans phrase.
        """

        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt.strip()
                    }
                ]
            )
            cluster_name = (
                response.get("message", {})
                .get("content", "")
                .strip()
            )
            if not cluster_name:
                cluster_name = "Thème non identifié"

        except Exception as e:
            print(f"Erreur cluster {cluster_id} : {e}")
            cluster_name = "Erreur génération"

        cluster_names.append(cluster_name)

    return cluster_names


# chunks = main()
# print(chunks)
# main()
new_chunk = recuperer_tous_les_vecteurs()
cluster = clustering(new_chunk)
# print("test")
cluster_name = generate_cluster_names(cluster)
print(cluster_name)



