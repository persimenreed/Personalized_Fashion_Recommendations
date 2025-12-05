#!/usr/bin/env python3
"""
Candidate diagnostics:
Per-week candidate statistics (Table 1) + per-week coverage.
"""

import gc
import re
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
INPUT_DIR = DATA_DIR / "input_data"
OUTPUT_DIR = DATA_DIR / "outputs" / "candidates"

TX_PATH = INPUT_DIR / "transactions_train.csv"

LABELS_PARQUET = OUTPUT_DIR / "labels_last_week.parquet"
LABELS_SET = OUTPUT_DIR / "labels_last_week.pkl"

# Weeks used in your project
WEEKS = [
    "2020-08-19",
    "2020-08-26",
    "2020-09-02",
    "2020-09-09",
    "2020-09-16",
    "2020-09-22",
]

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def hex16_to_int(s: str):
    return np.int64(np.uint64(int(s[-16:], 16)))


def build_labels():
    """Build the last-week positive labels (7-day window before max(t_dat))."""
    print("[labels] Building label cache...")
    tx = pd.read_csv(
        TX_PATH,
        usecols=["t_dat", "customer_id", "article_id"],
        dtype={"t_dat": "string", "customer_id": "string", "article_id": "int32"},
    )
    tx["customer_id"] = tx["customer_id"].str[-16:].apply(hex16_to_int)
    tx["t_dat"] = pd.to_datetime(tx["t_dat"])

    last_ts = tx["t_dat"].max()
    cut_ts = last_ts - pd.Timedelta(days=7)

    labels = tx[(tx["t_dat"] > cut_ts) & (tx["t_dat"] <= last_ts)]
    labels = labels[["customer_id", "article_id"]].drop_duplicates()
    labels["customer_id"] = labels["customer_id"].astype("int64")
    labels["article_id"] = labels["article_id"].astype("int32")

    labels.to_parquet(LABELS_PARQUET, index=False)
    with open(LABELS_SET, "wb") as f:
        pickle.dump(set(map(tuple, labels.to_numpy())), f)

    print(f"[labels] Saved {len(labels)} label pairs.")
    del tx, labels
    gc.collect()


def ensure_labels():
    """Ensure label parquet and pickle exist, then load them."""
    if not LABELS_PARQUET.exists() or not LABELS_SET.exists():
        build_labels()

    labels = pd.read_parquet(LABELS_PARQUET)
    with open(LABELS_SET, "rb") as f:
        label_set = pickle.load(f)

    return labels, label_set


def read_candidate(path: Path):
    df = pd.read_parquet(path, columns=["customer_id", "article_id"])
    df["customer_id"] = df["customer_id"].astype("int64")
    df["article_id"] = df["article_id"].astype("int32")
    return df


def coverage(label_set, df):
    arr = df[["customer_id", "article_id"]].drop_duplicates().to_numpy()
    return sum((cid, art) in label_set for cid, art in arr)


# ---------------------------------------------------------
# Per-week candidate statistics + coverage
# ---------------------------------------------------------

def find_week_files(week: str):
    """Returns all candidate files matching *_<week>.parquet."""
    pattern = re.compile(f"candidates_.*_{week}\\.parquet")
    return [f for f in OUTPUT_DIR.glob("*.parquet") if pattern.fullmatch(f.name)]


def compute_week_stats(week: str, total_labels, label_set):
    files = find_week_files(week)
    if not files:
        print(f"[WARN] No candidate files for week {week}")
        return None

    dfs = [read_candidate(f) for f in files]
    full = pd.concat(dfs).drop_duplicates()
    del dfs
    gc.collect()

    per_user = full.groupby("customer_id").size().values

    cov = coverage(label_set, full)
    recall = cov / total_labels if total_labels else 0.0

    stats = {
        "week": week,
        "candidates": len(full),
        "mean": float(np.mean(per_user)),
        "median": float(np.median(per_user)),
        "min": int(np.min(per_user)),
        "max": int(np.max(per_user)),
        "covered": cov,
        "recall": recall,
    }

    print(f"[week {week}] {stats}")
    return stats


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print("=== Candidate Diagnostics ===")

    # Load labels (7-day final window)
    labels_df, label_set = ensure_labels()
    total_labels = len(labels_df)
    print(f"[INFO] labels: {total_labels}, users: {labels_df.customer_id.nunique()}")

    # -----------------------
    # Per-week stats
    # -----------------------
    print("\n=== WEEKLY STATS (Table 1) ===")
    weekly_stats = [
        compute_week_stats(w, total_labels, label_set)
        for w in WEEKS
    ]

    with open(OUTPUT_DIR / "weekly_candidate_stats.json", "w") as f:
        json.dump(weekly_stats, f, indent=4)

    print("\nDone.")


if __name__ == "__main__":
    main()
