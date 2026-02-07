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

    vectors = np.array(vectors)
    texts = [chunk["text"] for chunk in chunks]

    dbscan = DBSCAN(
        eps=0.2,
        min_samples=2,
        metric="cosine"
    )
    labels = dbscan.fit_predict(vectors)

    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append({
            "text": texts[idx],
            "vector": vectors[idx],
            "cluster": label
        })

    noise_chunks = clusters.get(-1, [])
    strong_clusters = {k: v for k, v in clusters.items() if k != -1}

    final_clusters = {}
    current_label = 0

    for chunks in strong_clusters.values():
        final_clusters[current_label] = chunks
        current_label += 1

    if len(noise_chunks) >= 5:
        noise_vectors = np.array([c["vector"] for c in noise_chunks])

        k = int(np.sqrt(len(noise_chunks)))
        k = max(5, min(k, 30))

        kmeans = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=42
        )

        kmeans_labels = kmeans.fit_predict(noise_vectors)

        kmeans_groups = defaultdict(list)
        for idx, label in enumerate(kmeans_labels):
            kmeans_groups[label].append(noise_chunks[idx])

        for chunks in kmeans_groups.values():
            final_clusters[current_label] = chunks
            current_label += 1

    for cid, chunks in final_clusters.items():
        print(f"\n=== CLUSTER {cid} ({len(chunks)} chunks) ===")
        print(chunks[0]["vector"])

    return final_clusters
