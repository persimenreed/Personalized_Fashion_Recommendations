#!/usr/bin/env python3
"""
Minimal fast recall checker.
Run: python check_cover.py
If labels cache missing it is built (parquet + pickle set).
Outputs per-candidate recall and overall union recall.
"""

import gc
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
INPUT_DIR = DATA_DIR / "input_data"
OUTPUT_DIR = DATA_DIR / "outputs"

TX_PATH = INPUT_DIR / "transactions_train.csv"
LABELS_PARQUET = OUTPUT_DIR / "labels_last_week.parquet"
LABELS_CSV = OUTPUT_DIR / "labels_last_week.csv"
LABELS_SET = OUTPUT_DIR / "labels_last_week.pkl"

CANDIDATE_FILES = [
    #OUTPUT_DIR / "candidates_weekly_trending.parquet",
    #OUTPUT_DIR / "candidates_itemcf.parquet",
    #OUTPUT_DIR / "candidates_popularity.parquet",
    #OUTPUT_DIR / "candidates_recent_top.parquet",
    #OUTPUT_DIR / "candidates_repurchase.parquet",
    #OUTPUT_DIR / "candidates_user_overlap.parquet",
    #OUTPUT_DIR / "candidates_age_bucket_pop.parquet",
    #OUTPUT_DIR / "candidates_category_affinity.parquet",
    #OUTPUT_DIR / "candidates_same_product.parquet",
    #OUTPUT_DIR / "candidates_embedding.parquet",
]


def build_labels():
    print("[labels] build")
    tx = pd.read_csv(
        TX_PATH,
        usecols=["t_dat", "customer_id", "article_id"],
        dtype={"t_dat": "string", "customer_id": "string", "article_id": "int32"},
    )
    tx["customer_id"] = tx["customer_id"].str[-16:].apply(lambda h: np.int64(np.uint64(int(h, 16))))
    tx["t_dat"] = pd.to_datetime(tx["t_dat"])
    cut_ts = tx["t_dat"].max() - pd.Timedelta(days=7)
    labels = tx[(tx["t_dat"] > cut_ts) & (tx["t_dat"] <= tx["t_dat"].max())][["customer_id", "article_id"]]
    labels = labels.drop_duplicates().reset_index(drop=True)
    labels["customer_id"] = labels["customer_id"].astype("int64")
    labels["article_id"] = labels["article_id"].astype("int32")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(LABELS_PARQUET, index=False)
    labels.to_csv(LABELS_CSV, index=False)
    label_set = set(map(tuple, labels.to_numpy()))
    with open(LABELS_SET, "wb") as f:
        pickle.dump(label_set, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[build] cached parquet/csv/pickle in {OUTPUT_DIR}")
    del tx, labels
    gc.collect()


def ensure_labels():
    if not LABELS_PARQUET.exists() or not LABELS_SET.exists():
        build_labels()
    labels = pd.read_parquet(LABELS_PARQUET, columns=["customer_id", "article_id"])
    labels["customer_id"] = labels["customer_id"].astype("int64")
    labels["article_id"] = labels["article_id"].astype("int32")

    with open(LABELS_SET, "rb") as f:
        label_set = pickle.load(f)

    return labels, label_set


def read_candidate(path: Path):
    df = pd.read_parquet(path, columns=["customer_id", "article_id"])
    df["customer_id"] = df["customer_id"].astype("int64")
    df["article_id"] = df["article_id"].astype("int32")
    return df


def coverage_count(label_set, cand_df):
    arr = cand_df[["customer_id", "article_id"]].drop_duplicates().to_numpy()
    return sum((cid, art) in label_set for cid, art in arr)


def main():
    print("=== Candidate coverage check ===")
    labels_df, label_set = ensure_labels()
    total_labels = len(labels_df)
    total_users = labels_df["customer_id"].nunique()
    print(f"[info] total positive pairs: {total_labels} | unique users: {total_users}")

    # collect existing candidate files, report missing
    candidate_paths = []
    for path in CANDIDATE_FILES:
        if path.exists():
            candidate_paths.append(path)
        else:
            print(f"[miss] {path}")

    if not candidate_paths:
        print("[warn] no candidate parquet files found")
        return

    union_enabled = len(candidate_paths) > 1
    if union_enabled:
        union_set = set()

    for path in candidate_paths:
        cand = read_candidate(path)
        covered = coverage_count(label_set, cand)
        recall = covered / total_labels if total_labels else 0.0
        cand_users = cand["customer_id"].nunique()
        print(
            f"[src] {path.name:35s} | rows:{len(cand):9d} | covered:{covered:7d} | "
            f"recall:{recall:.4f} | users:{cand_users:7d}"
        )
        if union_enabled:
            union_set.update(map(tuple, cand[["customer_id", "article_id"]].to_numpy()))
        del cand
        gc.collect()

    if union_enabled:
        union_covered = sum(pair in label_set for pair in union_set)
        union_recall = union_covered / total_labels if total_labels else 0.0
        union_users = len({cid for cid, _ in union_set})
        print(f"[union] pairs:{len(union_set)} | covered:{union_covered} | recall:{union_recall:.4f} | users:{union_users}")


if __name__ == "__main__":
    main()