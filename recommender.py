"""
recommender.py

Scores catalog products against a user's selection history using cosine
similarity, then blends in a trend score so currently popular styles get a
small boost without overriding personal preference.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_user_profile_vector(feature_matrix, df, selected_product_ids):
    """
    Builds a user "taste profile" vector by averaging the feature vectors
    of their previously selected products.
    """
    indices = df.index[df["product_id"].isin(selected_product_ids)].tolist()
    if not indices:
        raise ValueError("None of the selected_product_ids were found in the catalog.")
    user_vector = feature_matrix[indices].mean(axis=0)
    return np.asarray(user_vector)


def compute_trend_scores(df, selection_counts=None):
    """
    selection_counts: optional dict {product_id: times_selected_recently}.
    If not provided, falls back to a neutral trend score of 0 for all items
    (i.e. no trend boost) so the pipeline still runs without real usage data.
    """
    if selection_counts is None:
        return {pid: 0.0 for pid in df["product_id"]}

    max_count = max(selection_counts.values()) if selection_counts else 1
    return {pid: selection_counts.get(pid, 0) / max_count for pid in df["product_id"]}


def recommend_items(feature_matrix, df, selected_product_ids, top_n=5,
                     selection_counts=None, trend_weight=0.15, exclude_selected=True):
    """
    Returns a DataFrame of the top_n recommended products, ranked by a
    blended score: (1 - trend_weight) * similarity + trend_weight * trend_score
    """
    user_vector = compute_user_profile_vector(feature_matrix, df, selected_product_ids)
    similarities = cosine_similarity(user_vector, feature_matrix).flatten()

    trend_scores = compute_trend_scores(df, selection_counts)
    trend_array = df["product_id"].map(trend_scores).to_numpy()

    blended_score = (1 - trend_weight) * similarities + trend_weight * trend_array

    result_df = df.copy()
    result_df["similarity"] = similarities
    result_df["trend_score"] = trend_array
    result_df["score"] = blended_score

    if exclude_selected:
        result_df = result_df[~result_df["product_id"].isin(selected_product_ids)]

    return result_df.sort_values("score", ascending=False).head(top_n)
