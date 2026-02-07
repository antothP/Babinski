import os
from pathlib import Path
from PyPDF2 import PdfReader
from chunker import chunkeriser_texte
from embeddings import get_embeddings
import numpy as np

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
        print(f"{pdf_path.name}: {len(chunks)} chunks")
        tous_les_chunks.extend(chunks)
    embeddings = get_embeddings(tous_les_chunks)
    numpy_array = np.array(embeddings, dtype=np.float32)
    return numpy_array

chunks = main()
print(chunks)
