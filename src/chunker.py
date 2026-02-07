from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunkeriser_texte(texte, chunk_size=500, overlap=50):
    """
    Découpe le texte en chunks optimisés.
    
    Args:
        texte: Texte à découper
        chunk_size: Taille max d'un chunk (défaut: 500)
        overlap: Chevauchement entre chunks (défaut: 50)
    
    Returns:
        Liste de chunks (strings)
    """
    texte = texte.strip()
    texte = ' '.join(texte.split())    
    separateurs = [
        "\n\n",
        "\n",
        ". ",
        "! ",
        "? ",
        "; ",
        ", ",
        " ",
        ""
    ]    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=separateurs,
        length_function=len
    )
    chunks = splitter.split_text(texte)
    chunks_filtres = [c for c in chunks if len(c) >= 30]    
    return chunks_filtres


if __name__ == "__main__":
    texte_test = """
    Ceci est un paragraphe de test pour vérifier le chunkeriser.
    Il contient plusieurs phrases. Et encore une autre phrase!
    
    Voici un nouveau paragraphe qui devrait être séparé.
    On teste le découpage intelligent du texte.
    """
    
    chunks = chunkeriser_texte(texte_test, chunk_size=100, overlap=20)
    
    print(f"Chunks créés: {len(chunks)}\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"[Chunk {i}] ({len(chunk)} car.)")
        print(chunk)
        print()