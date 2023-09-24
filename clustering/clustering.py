import hdbscan
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    # Load data
    embeddings = np.load(r'data\top_cluster_embeddings.npy')
    return embeddings

def get_clusters(embeddings):
    # Get clusters
    umap_embeddings = umap.UMAP(
        n_neighbors=15,
        n_components=15,
        metric='cosine'
        ).fit_transform(embeddings)
    
    cluster = hdbscan.HDBSCAN(
        min_cluster_size=30,
        metric='euclidean',
        cluster_selection_method='eom'
        ).fit(umap_embeddings)
    
    return cluster.labels_

def get_2d_data_for_plotting(embeddings):
    # Get 2D data for plotting
    umap_embeddings = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        metric='cosine'
        ).fit_transform(embeddings)
    
    return umap_embeddings

def plot_clusters(embeddings, cluster_labels):
    umap_data = get_2d_data_for_plotting(embeddings)
    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    result['labels'] = cluster_labels

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(20, 10))
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
    plt.colorbar()
    plt.savefig(r'plots\clusters.png', dpi=300)

def main():
    embeddings = load_data()
    cluster_labels = get_clusters(embeddings)
    plot_clusters(embeddings, cluster_labels)

if __name__ == '__main__':
    main()