# DBSCAN - Méthode par densité
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv('data_top_100.csv')

def apply_dbscan():
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    dbscan = DBSCAN(eps=1, min_samples=5)
    cluster_labels = dbscan.fit_predict(data_scaled)

    data['cluster_dbscan'] = cluster_labels

    plt.figure(figsize=(8, 5))
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=cluster_labels, cmap='viridis', marker='o')
    plt.title('Clusters - DBSCAN')
    plt.xlabel('Composante jours')
    plt.ylabel('Composante ventes')
    plt.show()

    return data
