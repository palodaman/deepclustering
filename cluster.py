from sklearn.cluster import KMeans
import sklearn.metrics as metrics

def cluster_embeddings(embeddings, n_clusters):
    kmeans = KMeans(n_clusters = n_clusters, n_init = 'auto')
    labels = kmeans.fit_predict(embeddings)
    return labels

def evaluate_cluster(labels_true, labels_pred):
    ari = metrics.adjusted_rand_score(labels_true, labels_pred)
    nmi = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    return {"ARI": ari, "NMI": nmi}