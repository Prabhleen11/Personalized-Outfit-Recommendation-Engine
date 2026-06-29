"""
main.py

End-to-end demo pipeline: load catalog -> build features -> cluster ->
score against user history -> assemble outfit recommendations.

Usage:
    python main.py --user_history "tshirt_001,jeans_004" --top_n 3
"""

import argparse

from feature_engineering import load_catalog, build_feature_matrix
from style_clustering import fit_style_clusters, label_clusters_by_dominant_occasion
from recommender import recommend_items
from outfit_assembler import assemble_outfits


def parse_args():
    parser = argparse.ArgumentParser(description="Outfit recommendation demo")
    parser.add_argument(
        "--catalog_path", type=str, default="data/sample_products.csv",
        help="Path to product catalog CSV"
    )
    parser.add_argument(
        "--user_history", type=str, required=True,
        help="Comma-separated product_ids the user previously selected"
    )
    parser.add_argument("--top_n", type=int, default=3, help="Number of outfits to recommend")
    parser.add_argument("--n_clusters", type=int, default=4, help="Number of style clusters")
    return parser.parse_args()


def main():
    args = parse_args()
    selected_ids = [pid.strip() for pid in args.user_history.split(",")]

    df = load_catalog(args.catalog_path)
    feature_matrix, _encoders = build_feature_matrix(df)

    _kmeans, cluster_labels = fit_style_clusters(feature_matrix, n_clusters=args.n_clusters)
    df_with_clusters, cluster_names = label_clusters_by_dominant_occasion(df, cluster_labels)

    candidates = recommend_items(
        feature_matrix, df_with_clusters, selected_ids,
        top_n=10,  # pull more candidates than outfits needed, for assembly
    )

    outfits = assemble_outfits(candidates, max_outfits=args.top_n)

    print(f"\nUser history: {selected_ids}\n")
    print("Recommended Outfits:")
    for i, outfit in enumerate(outfits, start=1):
        parts = [outfit["top"], outfit["bottom"]]
        if outfit["footwear"]:
            parts.append(outfit["footwear"])
        if outfit["accessory"]:
            parts.append(outfit["accessory"])
        print(f"  {i}. {' + '.join(parts)}   (style: {outfit['cluster']}, score: {outfit['avg_score']:.2f})")


if __name__ == "__main__":
    main()
