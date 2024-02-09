# Kmeans - Méthode Centroide
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv('data_top_100.csv')

def apply_kmeans():
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    num_clusters = 2
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_scaled)

    data['cluster_kmeans'] = cluster_labels

    plt.figure(figsize=(8, 5))
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=cluster_labels, cmap='viridis', marker='o')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='x')
    plt.title('Clusters - K-means')
    plt.xlabel('Composante jours')
    plt.ylabel('Composante ventes')
    plt.show()

    return data
