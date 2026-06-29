[README.md](https://github.com/user-attachments/files/29454754/README.md)
# Personalized-Outfit-Recommendation-Engine
A recommendation engine that suggests personalized outfit pairings based on user preferences and past selections. Uses K-Means clustering for style grouping and cosine similarity scoring with trend weighting, built with Python, Scikit-learn, and Pandas.

## Overview

A recommendation engine prototype that suggests outfit pairings based on a
user's style preferences, past selections, and currently trending styles.
The goal was to move beyond generic "best-seller" recommendations and give
users outfit suggestions that actually match their personal style profile.

## Problem Statement

Most e-commerce recommendation widgets suggest products in isolation
("customers who bought this also bought..."), but shoppers often think in
terms of *outfits* — a top with a bottom, paired with accessories. This
project explores a lightweight way to group products by style attributes
and recommend coherent outfit pairings personalized to a user's history.

## Approach

1. **Feature engineering** — Each product is represented by a feature vector
   built from style attributes (color family, pattern, category, occasion
   tag, fabric type) using one-hot encoding and TF-IDF on text descriptions.
2. **Clustering** — K-Means clustering groups products into style clusters
   (e.g. "casual minimal", "ethnic festive", "athleisure") so that
   recommendations stay stylistically coherent.
3. **Similarity scoring** — Cosine similarity between a user's previously
   selected items and the product catalog surfaces items that match their
   taste, weighted by how recently/often an item type was chosen.
4. **Trend weighting** — A simple popularity/trend score (based on recent
   selection frequency across users) nudges recommendations toward
   currently trending styles without overriding personal preference.
5. **Outfit assembly** — Once candidate items are scored, a simple pairing
   rule (top + bottom + optional accessory, matched by cluster and
   complementary color rules) assembles them into outfit sets rather than
   single-item suggestions.

## Tech Stack

- **Language:** Python
- **ML:** Scikit-learn (TF-IDF, K-Means, cosine similarity)
- **Data Handling:** Pandas, NumPy
- **Environment:** Google Colab

## Project Structure

```
outfit-recommender/
├── README.md
├── requirements.txt
├── feature_engineering.py     # Builds product feature vectors
├── style_clustering.py        # K-Means clustering into style groups
├── recommender.py             # Similarity scoring + trend weighting
├── outfit_assembler.py        # Pairs items into outfit sets
├── main.py                    # End-to-end pipeline / demo script
└── data/
    └── sample_products.csv    # Small sample product catalog for demo
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the end-to-end demo on the sample catalog
python main.py --user_history "tshirt_001,jeans_004" --top_n 3
```

## Example

```
Input:  User previously selected -> ["Casual White T-Shirt", "Blue Slim Jeans"]
Output: Recommended Outfits
  1. Casual White T-Shirt + Blue Slim Jeans + White Sneakers   (cluster: casual minimal)
  2. Striped Crop Top + Denim Shorts + Canvas Tote              (cluster: casual minimal)
  3. Graphic Tee + Mom-Fit Jeans + Bucket Hat                    (cluster: casual minimal, trending)
```

## Results

- Successfully grouped a sample product catalog into coherent style
  clusters using K-Means, validated qualitatively against expected style
  groupings (e.g. ethnic wear clustering separately from athleisure).
- Presented the working prototype to industry mentors from Myntra, who
  gave feedback on incorporating trend signals more strongly — incorporated
  as the trend-weighting step in the similarity scoring logic.

## Future Improvements

- Replace static one-hot/TF-IDF features with learned embeddings
  (e.g. CLIP-based image embeddings) for richer style similarity.
- Incorporate real-time trend signals from social/e-commerce data instead
  of a static frequency score.
- A/B test outfit-level recommendations against single-item recommendations
  to measure actual lift in engagement.

## Team

Built collaboratively as part of the Who-She Hackathon (Myntra),
a Women-in-Tech initiative.
