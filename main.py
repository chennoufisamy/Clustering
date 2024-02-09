# Fichier Main.py
import pandas as pd
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import DBSCAN
from ACP_kmeans import apply_ACP_kmeans
from kmeans_clustering import apply_kmeans
from cah_clustering import apply_hierarchical
from dbscan_clustering import apply_dbscan
from coudes_avec_Kmeans import apply_coudes
from silhouette_avec_kmeans import apply_silhouette

total_data = pd.read_csv('sales_train_validation.csv')

#Application ACP sur K-means (totalité des donées)
total_data = apply_ACP_kmeans(total_data)

#Application coudes sur K-means
data = apply_coudes()

#Application silhouette sur K-means
data = apply_silhouette()

# Application de K-means
data = apply_kmeans()
data_scaled_kmeans = data.drop(['cluster_kmeans'], axis=1)
db_kmeans = davies_bouldin_score(data_scaled_kmeans, data['cluster_kmeans'])
ch_kmeans = calinski_harabasz_score(data_scaled_kmeans, data['cluster_kmeans'])
print(f"Indice de Davies-Bouldin pour K-means : {db_kmeans}")
print(f"Indice de Calinski-Harabasz pour K-means : {ch_kmeans}")

# Application de CAH 
data = apply_hierarchical()
data_scaled_hierarchical = data.drop(['cluster_hierarchical'], axis=1)
db_hierarchical = davies_bouldin_score(data_scaled_hierarchical, data['cluster_hierarchical'])
ch_hierarchical = calinski_harabasz_score(data_scaled_hierarchical, data['cluster_hierarchical'])
print(f"Indice de Davies-Bouldin pour la méthode hiérarchique : {db_hierarchical}")
print(f"Indice de Calinski-Harabasz pour la méthode hiérarchique : {ch_hierarchical}")

# Application de DBSCAN
data = apply_dbscan()

# Vérification du nombre de clusters retournés par DBSCAN
num_clusters_dbscan = len(set(data['cluster_dbscan'])) - (1 if -1 in data['cluster_dbscan'] else 0)

if num_clusters_dbscan > 1:
    data_scaled_dbscan = data.drop(['cluster_kmeans', 'cluster_hierarchical', 'cluster_dbscan'], axis=1)
    db_dbscan = davies_bouldin_score(data_scaled_dbscan, data['cluster_dbscan'])
    ch_dbscan = calinski_harabasz_score(data_scaled_dbscan, data['cluster_dbscan'])
    print(f"Indice de Davies-Bouldin pour DBSCAN : {db_dbscan}")
    print(f"Indice de Calinski-Harabasz pour DBSCAN : {ch_dbscan}")
else:
    print("DBSCAN n'a trouvé qu'un seul cluster. Ajustez les paramètres si nécessaire.")
