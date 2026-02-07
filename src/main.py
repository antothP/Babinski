import os
from pathlib import Path
from PyPDF2 import PdfReader
from chunkeriser import chunkeriser_texte

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
            tous_les_chunks.append({'fichier': pdf_path.name, 'texte': chunk})
    return tous_les_chunks

chunks = main()
