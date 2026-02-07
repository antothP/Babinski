import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualisation_2d(data):
    embeddings = np.array(data).squeeze(axis=1)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    plt.figure(figsize=(12, 10))
    for i in range(len(coords)):
        plt.quiver(0, 0, coords[i, 0], coords[i, 1],
                   angles='xy', scale_units='xy', scale=1,
                   alpha=0.4, width=0.002)
    margin = 1.1
    max_range = max(np.abs(coords).max(), 1)
    plt.xlim(-max_range * margin, max_range * margin)
    plt.ylim(-max_range * margin, max_range * margin)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Visualisation vectorielle des embeddings (PCA)')
    plt.grid(True, alpha=0.3)
    plt.show()


def visualisation_2d_to_file(data, path):
    if not data:
        raise ValueError("Aucun vecteur à visualiser (données vides)")
    vectors = []
    for item in data:
        if isinstance(item, dict) and "vector" in item:
            v = item["vector"]
        else:
            v = item
        if isinstance(v, dict):
            v = next((x for x in v.values() if isinstance(x, (list, np.ndarray))), v)
        if hasattr(v, "tolist"):
            v = v.tolist()
        vectors.append(v)
    embeddings = np.array(vectors).squeeze()
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    pca = PCA(n_components=min(2, embeddings.shape[0], embeddings.shape[1]))
    coords = pca.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(12, 10))
    for i in range(len(coords)):
        ax.quiver(0, 0, coords[i, 0], coords[i, 1],
                  angles='xy', scale_units='xy', scale=1,
                  alpha=0.4, width=0.002, color='#333')
    margin = 1.1
    max_range = max(np.abs(coords).max(), 1)
    ax.set_xlim(-max_range * margin, max_range * margin)
    ax.set_ylim(-max_range * margin, max_range * margin)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Visualisation vectorielle des embeddings (PCA)')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    fig.savefig(path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)