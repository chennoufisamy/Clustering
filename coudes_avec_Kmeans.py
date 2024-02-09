import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data_top_100.csv')

def apply_coudes():
    # Standardiser les données
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Utiliser la méthode du coude pour trouver le nombre optimal de clusters
    inertias = []
    for k in range(1, 6):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_scaled)
        inertias.append(kmeans.inertia_)

    # Tracer la courbe du coude
    plt.plot(range(1, 6), inertias, marker='o')
    plt.title('Methode de coudes')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Inertie')
    plt.show()