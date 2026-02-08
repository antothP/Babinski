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


import re
from collections import Counter
from difflib import SequenceMatcher
import ollama

def extract_keywords(text, top_n=6):

    stopwords = {
        "le", "la", "les", "de", "des", "du", "un", "une", "et", "en",
        "à", "pour", "dans", "sur", "avec", "par", "est", "ce", "cette",
        "ces", "aux", "au", "ou", "plus", "moins"
    }

    words = re.findall(r"\b\w+\b", text.lower())

    words = [
        w for w in words
        if w not in stopwords and len(w) > 3
    ]

    return [w for w, _ in Counter(words).most_common(top_n)]

BAD_ENDINGS = {
    "de", "du", "des", "à", "et", "-", "pour", "en", "avec"
}


def clean_label(label, keywords):

    if not label:
        return " ".join(keywords[:3])
    label = re.sub(r"[^\w\s-]", "", label).strip()

    words = label.split()
    if len(words) < 3:
        return " ".join(keywords[:3])
    if words[-1].lower() in BAD_ENDINGS:
        words = words[:-1]

    words = words[:5]

    return " ".join(words)

def deduplicate_labels(labels, threshold=0.85):

    final = []

    for label in labels:

        duplicate = False

        for existing in final:
            score = SequenceMatcher(None, label, existing).ratio()

            if score > threshold:
                duplicate = True
                break

        if not duplicate:
            final.append(label)

    return final


def generate_cluster_names(final_clusters, model_name="gemma3"):

    cluster_names = []

    for cluster_id, chunks in final_clusters.items():

        texts = [
            c.get("text", "").strip()
            for c in chunks
            if c.get("text")
        ]

        full_text = " ".join(texts)

        if not full_text:
            cluster_names.append("Cluster vide")
            continue

        keywords = extract_keywords(full_text)

        prompt = f"""
Tu es expert en classification thématique IA.

Objectif :
Donner un label clair, complet et exploitable (3 à 5 mots).

Règles STRICTES :
- Aucun mot coupé
- Aucun mot incomplet
- Ne jamais finir par : de, du, à, et, -
- Pas de slogan
- Pas de marketing
- Pas de nom de marque
- Pas de phrase

Format attendu :
Nom + Complément + Objet

Exemples valides :
- Gouvernance des données IA
- Infrastructure cloud et GPU
- Formation des équipes IA
- Cas d’usage métiers IA
- Analyse avancée des données
- Sécurité et conformité IA

Extraits :
{full_text}

Mots-clés dominants :
{", ".join(keywords)}

Réponds uniquement par le label.
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

            raw_label = (
                response.get("message", {})
                .get("content", "")
                .strip()
            )

            final_label = clean_label(raw_label, keywords)

        except Exception as e:

            print(f"Erreur cluster {cluster_id} : {e}")

            final_label = " ".join(keywords[:3])

        cluster_names.append(final_label)

    cluster_names = deduplicate_labels(cluster_names)

    return cluster_names


print("test")
# cluster_name = generate_cluster_names(cluster)
# print(cluster_name)


