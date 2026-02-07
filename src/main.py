import os
from pathlib import Path
from PyPDF2 import PdfReader
from chunker import chunkeriser_texte
from embeddings import get_embeddings
from stockage import stocker_chunk, recherche_semantique, recuperer_tous_les_vecteurs, creer_schema
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


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


# chunks = main()
# print(chunks)
recuperer_tous_les_vecteurs()
