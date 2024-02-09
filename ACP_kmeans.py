import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy import stats

def apply_ACP_kmeans(df):
    # Supprimer les colonnes non désirées
    cols_to_drop = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    df.drop(cols_to_drop, axis=1, inplace=True)

    # Remplacer les valeurs manquantes par la moyenne de chaque série
    df.fillna(df.mean(), inplace=True)

    # Filtrer les outliers en utilisant le Z-score
    z = np.abs(stats.zscore(df))
    df = df[(z < 3).all(axis=1)]

    # Standardiser les données
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    # Appliquer l'ACP
    pca = PCA()
    data_pca = pca.fit_transform(data_scaled)

    # Utiliser le meilleur nombre de clusters trouvé par le score de silhouette
    best_n_clusters = 2

    # Initialiser l'algorithme K-means avec le nombre optimal de clusters
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=0)

    # Adapter l'algorithme K-means aux données réduites par ACP
    kmeans.fit(data_pca)

    # Récupérer les étiquettes de cluster pour chaque point de données
    cluster_labels = kmeans.labels_

    # Ajouter les étiquettes de cluster au DataFrame original pour une analyse ultérieure
    df['cluster'] = cluster_labels

    # Visualisation
    if data_pca.shape[1] >= 2:
        plt.figure(figsize=(8, 5))
        plt.scatter(data_pca[:, 0], data_pca[:, 1], c=cluster_labels, cmap='viridis', marker='o')
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='x')
        plt.title('Visualisation des clusters formés')
        plt.xlabel('Composante principale 1')
        plt.ylabel('Composante principale 2')
        plt.show()
