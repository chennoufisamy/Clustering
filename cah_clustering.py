# CAH - Classification Ascendente Hierarchique
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv('data_top_100.csv')

def apply_hierarchical():
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    agg = AgglomerativeClustering(n_clusters=2, linkage='ward')
    cluster_labels = agg.fit_predict(data_scaled)

    data['cluster_hierarchical'] = cluster_labels

    plt.figure(figsize=(8, 5))
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=cluster_labels, cmap='viridis', marker='o')
    plt.title('Clusters - Méthode hiérarchique')
    plt.xlabel('Composante jours')
    plt.ylabel('Composante ventes')
    plt.show()

    return data
