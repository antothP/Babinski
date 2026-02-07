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
    dbscan = DBSCAN(
        eps=0.2,
        min_samples=2,
        metric="cosine"
    )
    labels = dbscan.fit_predict(np_vectors)
    noisy_chunks = []
    final_clusters = defaultdict(list)

    for i, label in enumerate(labels):
        if label == -1:
            noisy_chunks.append(vectors[i])
        else:
            final_clusters[label].append(chunks[i])
    if (len(noisy_chunks) >= 5):
        k = int(np.sqrt(len(noisy_chunks)))
        kmeans = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=42
        )
        numpy_noisy = np.array(noisy_chunks)
        kmeans_labels = kmeans.fit_predict(numpy_noisy)
        print(kmeans_labels)
        kmeans_clusters = defaultdict(list)
        for i, label in enumerate(kmeans_labels):
            kmeans_clusters[label].append(noisy_chunks[i])
        max_cluster_id = max(final_clusters.keys()) if final_clusters else -1
        for i, cluster in enumerate(kmeans_clusters.values()):
            final_clusters[i + max_cluster_id + 1].extend(cluster)
    for id, chunk in final_clusters.items():
        print(f"Cluster {id} : {len(chunk)} chunks")

    return final_clusters

#voir ce que cest un RAG