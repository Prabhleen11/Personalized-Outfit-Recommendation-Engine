"""
feature_engineering.py

Builds numeric feature vectors for each product using one-hot encoding of
categorical style attributes (color, pattern, category, occasion) combined
with a TF-IDF representation of the free-text description.
"""

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder


def load_catalog(csv_path):
    return pd.read_csv(csv_path)


def build_feature_matrix(df):
    """
    Returns:
        feature_matrix: sparse matrix of shape (n_products, n_features)
        encoders: dict with fitted encoder/vectorizer objects (for reuse on new items)
    """
    categorical_cols = ["category", "color", "pattern", "occasion"]
    ohe = OneHotEncoder(handle_unknown="ignore")
    cat_features = ohe.fit_transform(df[categorical_cols])

    tfidf = TfidfVectorizer(max_features=200, stop_words="english")
    text_features = tfidf.fit_transform(df["description"].fillna(""))

    feature_matrix = hstack([cat_features, text_features]).tocsr()

    encoders = {"onehot": ohe, "tfidf": tfidf, "categorical_cols": categorical_cols}
    return feature_matrix, encoders


def transform_new_item(item_row, encoders):
    """
    Transforms a single new product (pd.Series) into the same feature space
    as the fitted catalog, using already-fitted encoders.
    """
    cat_df = pd.DataFrame([item_row[encoders["categorical_cols"]]])
    cat_features = encoders["onehot"].transform(cat_df)
    text_features = encoders["tfidf"].transform([item_row.get("description", "")])
    return hstack([cat_features, text_features]).tocsr()
