import numpy as np
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