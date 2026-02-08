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
        system_prompt = """Tu es un assistant intelligent qui répond aux questions en te basant sur le contexte fourni.
Utilise les informations présentes dans le contexte pour répondre.
Si l'information n'est pas dans les contextes utilise l'information qui s'y rapproche le plus sauf si le taux de similarité est faible.
Lorsque tu répond ne precise pas le contexte répond à la question sans parler de ce qu'est un chunks et
un contexte. Si tu dois parler de plusieurs mot clés decrit les"""
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
