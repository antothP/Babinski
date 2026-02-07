from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunkeriser_texte(texte, chunk_size=500, overlap=50):
    texte = texte.strip()
    texte = ' '.join(texte.split())
    separateurs = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separators=separateurs, length_function=len)
    chunks = splitter.split_text(texte)
    chunks_filtres = [c for c in chunks if len(c) >= 30]
    return chunks_filtres

texte_test = """Ceci est un paragraphe de test pour verifier le chunkeriser. Il contient plusieurs phrases. Et encore une autre phrase! Voici un nouveau paragraphe qui devrait etre separe. On teste le decoupage intelligent du texte."""
chunks = chunkeriser_texte(texte_test, chunk_size=100, overlap=20)
