import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from embeddings import get_embeddings

import json


client = weaviate.connect_to_local(
    host="localhost",
    port=8080
)

def creer_schema():
    try:
        if client.collections.exists("Chunk"):
            client.collections.delete("Chunk")
            print("Ancienne collection supprimée")

        client.collections.create(
            name="Chunk",
            vector_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="metadata", data_type=DataType.TEXT),
            ],
        )
        print("Schéma créé avec succès")
    except Exception as e:
        print(f"Erreur création schéma: {e}")

def stocker_chunk(chunk_text, metadata, embedding_vector):
    try:
        collection = client.collections.get("Chunk")
        vector = embedding_vector.tolist() if hasattr(embedding_vector, "tolist") else embedding_vector
        if len(vector) > 0 and isinstance(vector[0], (list, tuple)):
            vector = vector[0]

        if not isinstance(metadata, str):
            metadata = json.dumps(metadata, ensure_ascii=False)

        uuid = collection.data.insert(
            properties={"text": chunk_text, "metadata": metadata},
            vector=vector,
        )
        print(f"✅ Chunk stocké: {chunk_text[:50]}... (UUID: {uuid})")
        return uuid
    except Exception as e:
        print(f"Erreur stockage: {e}")
        return None

def recherche_semantique(query, top_k=5):
    try:
        collection = client.collections.get("Chunk")
        query_vector = get_embeddings(query)
        vector = query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector
        if len(vector) > 0 and isinstance(vector[0], (list, tuple)):
            vector = vector[0]
        response = collection.query.near_vector(
            near_vector=vector,
            limit=top_k,
            return_metadata=MetadataQuery(certainty=True)
        )
        resultats = []
        print(f"Trouvé {len(response.objects)} résultats")
        for obj in response.objects:
            resultats.append({
                "text": obj.properties.get("text"),
                "metadata": obj.properties.get("metadata"),
                "certainty": obj.metadata.certainty if obj.metadata else None
            })
        return resultats
    except Exception as e:
        print(f"Erreur recherche: {e}")
        return []

def recuperer_tous_les_vecteurs():
    try:
        collection = client.collections.get("Chunk")
        response = collection.query.fetch_objects(
            limit=10000,
            include_vector=True
        )
        chunks = []

        for i, obj in enumerate(response.objects):
            chunk_data = {
                "text": obj.properties.get("text"),
                "metadata": obj.properties.get("metadata"),
                "vector": obj.vector
            }
            chunks.append(chunk_data)
            
            # print(f"\n--- Chunk {i+1} ---")
            # print(f"Text: {chunk_data['text'][:100] if chunk_data['text'] else 'N/A'}...")
            # print(f"Metadata: {chunk_data['metadata']}")

            vector = chunk_data["vector"]
            # if isinstance(vector, list):
            #     print(f"Vector (5 premiers): {vector[:5]}...")
            # elif isinstance(vector, dict):
            #     for name, vec in vector.items():
            #         if isinstance(vec, list):
            #             print(f"Vector[{name}] (5 premiers): {vec[:5]}...")
            # else:
            #     print("Vector: N/A")
        
        return chunks
        
    except Exception as e:
        print(f"Erreur récupération: {e}")
        return []

def verifier_connexion():
    try:
        if client.is_ready():
            print("✅ Connexion Weaviate OK")
            collections = client.collections.list_all()
            print(f"Collections disponibles: {list(collections.keys())}")
            return True
        else:
            print("Weaviate n'est pas prêt")
            return False
    except Exception as e:
        print(f"Erreur connexion: {e}")
        return False

def fermer_connexion():
    client.close()
    print("👋 Connexion fermée")
