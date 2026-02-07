import os
import json
from pathlib import Path
from PyPDF2 import PdfReader
from chunker import chunkeriser_texte
from embeddings import get_embeddings
from visualisation import visualisation_2d

CACHE_FILE = "embeddings_cache.json"

def lire_pdf(fichier_path):
    texte = ""
    reader = PdfReader(fichier_path)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            texte += page_text + "\n"
    return texte

def main():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    dossier_data = "data"
    pdfs = list(Path(dossier_data).glob("*.pdf"))
    tous_les_chunks = []
    for pdf_path in pdfs:
        texte = lire_pdf(pdf_path)
        chunks = chunkeriser_texte(texte, chunk_size=500, overlap=50)
        print(chunks)
        for chunk in chunks:
            get_embeddings(chunk, tous_les_chunks)
    with open(CACHE_FILE, "w") as f:
        json.dump(tous_les_chunks, f)
    return tous_les_chunks

chunks = main()
print(chunks)
visualisation_2d(chunks)
