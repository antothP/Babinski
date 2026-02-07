import os
from pathlib import Path
from PyPDF2 import PdfReader
from chunkeriser import chunkeriser_texte


def lire_pdf(fichier_path):
    """Lit un PDF et retourne le texte."""
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
    print(f"📁 Fichiers trouvés: {len(pdfs)}")    
    tous_les_chunks = []    
    for pdf_path in pdfs:
        print(f"\n📖 Traitement: {pdf_path.name}")
        texte = lire_pdf(pdf_path)
        print(f"   Caractères extraits: {len(texte)}")        
        chunks = chunkeriser_texte(texte, chunk_size=500, overlap=50)
        print(f"   Chunks créés: {len(chunks)}")        
        for chunk in chunks:
            tous_les_chunks.append({
                'fichier': pdf_path.name,
                'texte': chunk
            })    
    print(f"\n✅ TOTAL: {len(tous_les_chunks)} chunks de {len(pdfs)} fichiers")    
    print("\n📋 Aperçu des premiers chunks:")
    for i, chunk in enumerate(tous_les_chunks[:3], 1):
        print(f"\n[Chunk {i}] - Fichier: {chunk['fichier']}")
        print(f"{chunk['texte'][:200]}...")
    
    return tous_les_chunks


if __name__ == "__main__":
    chunks = main()
    print(f"\n🎉 Array final contient {len(chunks)} chunks prêts à être utilisés")