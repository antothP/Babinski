import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from collections import defaultdict

def clustering(chunks):
    vectors = []
    for chunk in chunks:
        vector = chunk["vector"]
        if isinstance(vector, dict):
            vector = list(vector.values())[0]
        if hasattr(vector, 'tolist'):
            vector = vector.tolist()
        vectors.append(vector)
    
    np_vectors = np.array(vectors)
    
    # DBSCAN clustering
    dbscan = DBSCAN(
        eps=0.2,
        min_samples=2,
        metric="cosine"
    )
    labels = dbscan.fit_predict(np_vectors)
    
    noisy_chunks = []
    noisy_indices = []  # Garder trace des indices originaux
    final_clusters = defaultdict(list)
    
    for i, label in enumerate(labels):
        if label == -1:
            noisy_chunks.append(vectors[i])
            noisy_indices.append(i)  # Stocker l'index original
        else:
            final_clusters[label].append(chunks[i])
    
    # KMeans pour les points bruités
    if len(noisy_chunks) >= 5:
        k = min(int(np.sqrt(len(noisy_chunks))), len(noisy_chunks))  # Limiter k
        kmeans = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=42
        )
        numpy_noisy = np.array(noisy_chunks)
        kmeans_labels = kmeans.fit_predict(numpy_noisy)
        
        max_cluster_id = max(final_clusters.keys()) if final_clusters else -1
        
        for i, label in enumerate(kmeans_labels):
            original_idx = noisy_indices[i]  # Utiliser l'index original
            cluster_id = label + max_cluster_id + 1
            final_clusters[cluster_id].append(chunks[original_idx])
    
    # Affichage
    for id, cluster in final_clusters.items():
        for c in cluster:
            print(f"Cluster {id}: {c['text'][:50] if c['text'] else 'N/A'}...")
    
    return final_clusters