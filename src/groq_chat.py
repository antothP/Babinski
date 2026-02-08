import os
from groq import Groq

def generer_reponse_groq(question, chunks_pertinents):
    """
    Génère une réponse avec Groq en utilisant les chunks pertinents

    Args:
        question: La question de l'utilisateur
        chunks_pertinents: Liste de dicts avec 'text', 'metadata', 'certainty'

    Returns:
        str: La réponse générée par Groq
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        contexte = "\n\n".join([
            f"[Chunk {i+1} - Similarité: {chunk.get('certainty', 'N/A')}]\n{chunk['text']}"
            for i, chunk in enumerate(chunks_pertinents)
        ])
        system_prompt = """Tu es un assistant intelligent chargé de répondre aux questions de l utilisateur en t appuyant en priorité sur les informations fournies dans le contexte.

Règles de réponse :
Utilise exclusivement les informations présentes dans le contexte lorsqu elles sont pertinentes.
Si une information exacte n est pas disponible, utilise celle qui s en rapproche le plus, uniquement si la similarité est suffisante.
Si la similarité est faible ou incertaine, indique que l information n est pas clairement établie.

Contraintes de formulation :
Ne mentionne jamais le contexte, les sources, les chunks ou leur existence.
Réponds directement à la question de manière naturelle et fluide.

Gestion des mots-clés :
Lorsque la réponse contient un ou plusieurs mots-clés importants, explique chaque mot-clé de façon clair et pédagogique.
Si plusieurs mots-clés sont présents, décris-les, sépare les simplement par des retour a la ligne de la maniere "[mot clé] : [explication] (retour à la ligne)"

Objectif :
Fournir une réponse fiable, claire et structurée, en privilégiant la précision et la compréhension.
"""
        user_prompt = f"""Contexte:
{contexte}

Question: {question}

Réponds à la question en te basant sur le contexte fourni ci-dessus."""
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1024,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Erreur Groq: {e}")
        return f"Erreur lors de la génération de la réponse: {str(e)}"
