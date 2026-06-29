"""
outfit_assembler.py

Takes scored candidate products and assembles them into outfit sets
(top + bottom + optional accessory/footwear), staying within the same
style cluster so pairings remain coherent.
"""

import pandas as pd


def assemble_outfits(candidates_df, max_outfits=3):
    """
    candidates_df must include columns: category, cluster (or cluster_name), score

    Greedily builds outfits by pairing the highest-scoring top with the
    highest-scoring bottom in the same cluster, optionally adding a
    footwear and accessory item from that same cluster.
    """
    outfits = []
    used_ids = set()

    cluster_col = "cluster_name" if "cluster_name" in candidates_df.columns else "cluster"

    for cluster_value, group in candidates_df.groupby(cluster_col):
        tops = group[group["category"] == "top"].sort_values("score", ascending=False)
        bottoms = group[group["category"] == "bottom"].sort_values("score", ascending=False)
        footwear = group[group["category"] == "footwear"].sort_values("score", ascending=False)
        accessories = group[group["category"] == "accessory"].sort_values("score", ascending=False)

        if tops.empty or bottoms.empty:
            continue

        top_item = tops.iloc[0]
        bottom_item = bottoms.iloc[0]

        outfit = {
            "cluster": cluster_value,
            "top": top_item["name"],
            "bottom": bottom_item["name"],
            "footwear": footwear.iloc[0]["name"] if not footwear.empty else None,
            "accessory": accessories.iloc[0]["name"] if not accessories.empty else None,
            "avg_score": float(pd.Series([
                top_item["score"], bottom_item["score"]
            ]).mean()),
        }
        outfits.append(outfit)

    outfits = sorted(outfits, key=lambda o: o["avg_score"], reverse=True)
    return outfits[:max_outfits]
