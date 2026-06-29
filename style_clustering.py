"""
style_clustering.py

Groups products into style clusters using K-Means on the product feature
matrix, so that recommendations and outfit pairings stay stylistically
coherent (e.g. "casual minimal" vs "ethnic festive" vs "athleisure").
"""

from sklearn.cluster import KMeans


def fit_style_clusters(feature_matrix, n_clusters=4, random_state=42):
    """
    Fits K-Means on the feature matrix and returns the fitted model
    along with cluster labels for each product.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(feature_matrix)
    return kmeans, cluster_labels


def label_clusters_by_dominant_occasion(df, cluster_labels):
    """
    Assigns a human-readable name to each cluster based on the most common
    'occasion' value of products in that cluster (purely for readability in
    demos/output — not used in the similarity math).
    """
    df = df.copy()
    df["cluster"] = cluster_labels

    cluster_names = {}
    for cluster_id, group in df.groupby("cluster"):
        dominant_occasion = group["occasion"].mode().iloc[0]
        cluster_names[cluster_id] = dominant_occasion

    df["cluster_name"] = df["cluster"].map(cluster_names)
    return df, cluster_names
