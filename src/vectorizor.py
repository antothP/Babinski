import requests
import json
import hashlib
import os


def _fallback_vector_from_text(text, dim=512):
    """Deterministic fallback: derive a float vector from a SHA256 stream of the text.

    Returns a list of floats in approx range [-1, 1]. Same input -> same vector.
    """
    # Start from the hash of the text, then expand by re-hashing to reach required dim
    vec = []
    src = hashlib.sha256(text.encode("utf-8")).digest()
    while len(vec) < dim:
        src = hashlib.sha256(src).digest()
        for b in src:
            if len(vec) >= dim:
                break
            # Map byte (0..255) to float roughly in [-1,1]
            vec.append((b - 128) / 128.0)
    return vec


def get_question_vector(question, model="nomic-embed-text", fallback_dim=512):
    """Return embedding for question using Ollama; if unavailable, return deterministic fallback.

    To disable the automatic fallback and receive None on failure, set env var DISABLE_FALLBACK=1.
    """
    url = "http://localhost:11434/api/embeddings"
    payload = {
        "model": model,
        "prompt": question
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            # Ollama returns {"embedding": [...]} in the original code
            data = response.json()
            if isinstance(data, dict) and "embedding" in data:
                return data["embedding"]
            # Sometimes API may return embedding directly
            return data
        else:
            print(f"Erreur Ollama: status={response.status_code} body={response.text}")
    except Exception as e:
        print(f"Erreur vectorisation (connexion Ollama): {e}")

    # Fallback behaviour
    if os.getenv("DISABLE_FALLBACK", "0") in ("1", "true", "True"):
        print("DISABLE_FALLBACK is set -> returning None instead of fallback vector")
        return None

    print("Ollama indisponible -> utilisation d'un vecteur factice de fallback pour continuer le dev")
    return _fallback_vector_from_text(question, dim=fallback_dim)
