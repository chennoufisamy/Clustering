import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data_top_100.csv')

def apply_silhouette():
    # Définir une plage de nombres de clusters à évaluer
    min_clusters = 2
    max_clusters = 10

    # Liste pour stocker les scores de silhouette
    silhouette_scores = []

    # Calculer le score de silhouette pour chaque nombre de clusters
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Afficher le graphe
    plt.figure(figsize=(10, 6))
    plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Score de silhouette pour différents nombres de clusters')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Score de silhouette moyen')
    plt.xticks(range(min_clusters, max_clusters + 1))
    plt.grid(True)
    plt.show()
