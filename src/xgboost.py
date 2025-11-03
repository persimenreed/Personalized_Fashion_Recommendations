import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta

from sklearn.metrics import average_precision_score
from xgboost import XGBRanker

# -----------------------
# Helpers
# -----------------------

def week_period(series):
    # Weekly buckets ending on Sunday to match Kaggle common practice
    return series.dt.to_period('W-SUN')


def read_transactions(data_dir: Path, start_date: str = None, end_date: str = None, usecols=None):
    tx_path = data_dir / "transactions_train.csv"
    if usecols is None:
        usecols = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]

    dtypes = {
        "customer_id": "string",
        "article_id": "int32",
        "price": "float32",
        "sales_channel_id": "int8",
    }

    parse_dates = ["t_dat"]
    df = pd.read_csv(tx_path, usecols=usecols, dtype=dtypes, parse_dates=parse_dates)
    if start_date:
        df = df[df["t_dat"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["t_dat"] <= pd.to_datetime(end_date)]
    return df


def read_articles(data_dir: Path):
    art_path = data_dir / "articles.csv"
    usecols = [
        "article_id",
        "product_type_no",
        "product_group_name",
        "index_code",
        "section_no",
        "graphical_appearance_no",
        "colour_group_code",
        "perceived_colour_value_id",
        "perceived_colour_master_id",
    ]
    dtypes = {
        "article_id": "int32",
        "product_type_no": "int16",
        "product_group_name": "string",
        "index_code": "string",
        "section_no": "int16",
        "graphical_appearance_no": "int16",
        "colour_group_code": "int16",
        "perceived_colour_value_id": "int16",
        "perceived_colour_master_id": "int16",
    }
    df = pd.read_csv(art_path, usecols=usecols, dtype=dtypes)
    # Convert some to categorical to leverage XGBoost categorical support
    cat_cols = ["product_group_name", "index_code"]
    for c in cat_cols:
        df[c] = df[c].astype("category")
    return df


def compute_popular_items(df_hist: pd.DataFrame, topk: int = 200):
    pop = (
        df_hist.groupby("article_id", as_index=False)["customer_id"]
        .count()
        .rename(columns={"customer_id": "cnt"})
        .sort_values("cnt", ascending=False)
    )
    return pop["article_id"].head(topk).tolist()


def window_counts(df: pd.DataFrame, key: str, cutoff: pd.Timestamp, days: int):
    mask = (df["t_dat"] > cutoff - timedelta(days=days)) & (df["t_dat"] <= cutoff)
    cnt = df.loc[mask].groupby(key).size().rename(f"{key}_cnt_{days}d")
    return cnt


def recency_last_date(df: pd.DataFrame, keys, cutoff: pd.Timestamp, name: str):
    last = df.groupby(keys)["t_dat"].max().rename(f"{name}_last_dt")
    rec = (cutoff - last).dt.days.rename(f"{name}_recency_days")
    return last, rec


def build_aggregates(df_hist: pd.DataFrame, cutoff: pd.Timestamp):
    # Item-level popularity and recency
    feats = []

    for key in ["article_id"]:
        c1 = window_counts(df_hist, key, cutoff, 7)
        c2 = window_counts(df_hist, key, cutoff, 14)
        c4 = window_counts(df_hist, key, cutoff, 28)
        c12 = window_counts(df_hist, key, cutoff, 84)
        base = pd.DataFrame(index=df_hist[key].unique())
        base = base.join([c1, c2, c4, c12], how="left").fillna(0)
        base = base.reset_index().rename(columns={"index": key})
        feats.append(base)

    item_feats = feats[0]
    item_last_dt, item_rec = recency_last_date(df_hist, ["article_id"], cutoff, "item")
    item_last_dt = item_last_dt.reset_index()
    item_rec = item_rec.reset_index()
    item_feats = (
        item_feats.merge(item_last_dt, on="article_id", how="left")
        .merge(item_rec, on="article_id", how="left")
    )

    # Item average price
    item_price = (
        df_hist.groupby("article_id")["price"].mean().astype("float32").rename("item_avg_price_84d")
    )
    item_feats = item_feats.merge(item_price.reset_index(), on="article_id", how="left")

    # User-level stats
    u1 = window_counts(df_hist, "customer_id", cutoff, 7)
    u2 = window_counts(df_hist, "customer_id", cutoff, 14)
    u4 = window_counts(df_hist, "customer_id", cutoff, 28)
    u12 = window_counts(df_hist, "customer_id", cutoff, 84)
    user_feats = pd.DataFrame(index=df_hist["customer_id"].unique())
    user_feats = user_feats.join([u1, u2, u4, u12], how="left").fillna(0).reset_index().rename(
        columns={"index": "customer_id"}
    )
    user_last_dt, user_rec = recency_last_date(df_hist, ["customer_id"], cutoff, "user")
    user_feats = (
        user_feats.merge(user_last_dt.reset_index(), on="customer_id", how="left")
        .merge(user_rec.reset_index(), on="customer_id", how="left")
    )
    user_price = (
        df_hist.groupby("customer_id")["price"].mean().astype("float32").rename("user_avg_price_84d")
    )
    user_feats = user_feats.merge(user_price.reset_index(), on="customer_id", how="left")

    # User-item last interaction and count (over history window)
    ui_last = (
        df_hist.sort_values("t_dat")
        .groupby(["customer_id", "article_id"], as_index=False)
        .agg(last_dt=("t_dat", "max"), ui_cnt=("t_dat", "count"))
    )
    ui_last["ui_recency_days"] = (cutoff - ui_last["last_dt"]).dt.days.astype("float32")

    return item_feats, user_feats, ui_last


def generate_candidates(df_hist: pd.DataFrame,
                        target_df: pd.DataFrame,
                        cutoff: pd.Timestamp,
                        articles_popular_last4w,
                        negatives_per_user: int = 80):
    # Positives: items purchased in the target week
    positives = target_df[["customer_id", "article_id"]].drop_duplicates()
    positives["label"] = 1

    # Rebuy candidates: items user bought in last 4 weeks before cutoff
    mask_rebuy = (df_hist["t_dat"] > cutoff - timedelta(days=28)) & (df_hist["t_dat"] <= cutoff)
    recent = df_hist.loc[mask_rebuy, ["customer_id", "article_id"]].drop_duplicates()
    recent["src"] = "rebuy"

    # Global popular negatives
    pop_items = set(articles_popular_last4w)

    # Build negative samples per user
    users = target_df["customer_id"].unique()
    neg_rows = []
    for u in users:
        # already-positive to exclude
        pos_u = set(positives.loc[positives["customer_id"] == u, "article_id"].tolist())
        # recent items also included as personalized candidates
        recent_u = set(recent.loc[recent["customer_id"] == u, "article_id"].tolist())
        base_candidates = list((pop_items | recent_u) - pos_u)
        if not base_candidates:
            continue
        take = base_candidates[:negatives_per_user] if len(base_candidates) > negatives_per_user else base_candidates
        neg_rows.extend([(u, a, 0) for a in take])

    negatives = pd.DataFrame(neg_rows, columns=["customer_id", "article_id", "label"])

    # Union candidates
    candidates = pd.concat(
        [positives, negatives], axis=0, ignore_index=True
    ).drop_duplicates(subset=["customer_id", "article_id"], keep="first")

    # Ensure only users present in target_df
    candidates = candidates[candidates["customer_id"].isin(users)]
    return candidates


def map_at_k_per_user(y_true, y_score, k=12):
    # y_true/y_score are arrays aligned by user; caller should compute per-user
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order][:k]
    precisions = []
    correct = 0
    for i, rel in enumerate(y_true_sorted, 1):
        if rel == 1:
            correct += 1
            precisions.append(correct / i)
    return float(np.mean(precisions)) if precisions else 0.0


def evaluate_map12(df_pred: pd.DataFrame):
    # df_pred: columns [customer_id, article_id, label, pred]
    scores = []
    for u, g in df_pred.groupby("customer_id"):
        y_true = g["label"].astype(int).values
        y_score = g["pred"].astype(float).values
        scores.append(map_at_k_per_user(y_true, y_score, k=12))
    return float(np.mean(scores)) if scores else 0.0


# -----------------------
# Main pipeline
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="H&M XGBoost Ranker baseline")
    parser.add_argument("--data_dir", type=str, default="data/h-and-m-personalized-fashion-recommendations")
    parser.add_argument("--output_dir", type=str, default="outputs/xgboost_ranker")
    parser.add_argument("--start_date", type=str, default="2020-06-01", help="Trim transactions from this date")
    parser.add_argument("--weeks_history", type=int, default=16, help="Cap history window (in weeks) from last date")
    parser.add_argument("--negatives_per_user", type=int, default=80)
    parser.add_argument("--top_popular", type=int, default=200)
    parser.add_argument("--n_estimators", type=int, default=600)
    parser.add_argument("--learning_rate", type=float, default=0.07)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--sample", action="store_true", help="Speed mode: fewer users/items for quick run")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading articles...")
    articles = read_articles(data_dir)

    print("Loading transactions (trimmed)...")
    tx = read_transactions(data_dir, start_date=args.start_date)

    # Optional: reduce to last N weeks relative to dataset max date
    last_date = tx["t_dat"].max()
    if args.weeks_history is not None and args.weeks_history > 0:
        cutoff_hist = last_date - pd.Timedelta(weeks=args.weeks_history)
        tx = tx[tx["t_dat"] >= cutoff_hist].reset_index(drop=True)

    # Create weekly period
    tx["week"] = week_period(tx["t_dat"])
    last_week = tx["week"].max()
    train_week = last_week - 1
    valid_week = last_week

    # Historical frame for features up to day before target weeks
    valid_start = valid_week.start_time
    train_start = train_week.start_time
    train_cutoff = train_start - pd.Timedelta(days=1)
    valid_cutoff = valid_start - pd.Timedelta(days=1)

    # Split frames
    df_hist_train = tx[tx["t_dat"] <= train_cutoff].copy()
    df_hist_valid = tx[tx["t_dat"] <= valid_cutoff].copy()
    df_target_train = tx[(tx["t_dat"] >= train_week.start_time) & (tx["t_dat"] <= train_week.end_time)].copy()
    df_target_valid = tx[(tx["t_dat"] >= valid_week.start_time) & (tx["t_dat"] <= valid_week.end_time)].copy()

    if args.sample:
        # Focus on users that bought in the last two weeks; sample subset for speed
        users_keep = (
            tx[tx["week"].isin([train_week, valid_week])]["customer_id"].drop_duplicates().sample(
                frac=0.35, random_state=42
            )
        )
        df_hist_train = df_hist_train[df_hist_train["customer_id"].isin(users_keep)]
        df_hist_valid = df_hist_valid[df_hist_valid["customer_id"].isin(users_keep)]
        df_target_train = df_target_train[df_target_train["customer_id"].isin(users_keep)]
        df_target_valid = df_target_valid[df_target_valid["customer_id"].isin(users_keep)]

    print("Computing popularity for negatives...")
    pop4_train = compute_popular_items(
        df_hist_train[df_hist_train["t_dat"] > (train_cutoff - pd.Timedelta(days=28))], topk=args.top_popular
    )
    pop4_valid = compute_popular_items(
        df_hist_valid[df_hist_valid["t_dat"] > (valid_cutoff - pd.Timedelta(days=28))], topk=args.top_popular
    )

    print("Generating candidates (train)...")
    cand_train = generate_candidates(
        df_hist_train, df_target_train, train_cutoff, pop4_train, negatives_per_user=args.negatives_per_user
    )
    print(f"Train candidates: {len(cand_train):,}")

    print("Generating candidates (valid)...")
    cand_valid = generate_candidates(
        df_hist_valid, df_target_valid, valid_cutoff, pop4_valid, negatives_per_user=args.negatives_per_user
    )
    print(f"Valid candidates: {len(cand_valid):,}")

    print("Building aggregates (train cutoff)...")
    item_train, user_train, ui_train = build_aggregates(df_hist_train, train_cutoff)
    print("Building aggregates (valid cutoff)...")
    item_valid, user_valid, ui_valid = build_aggregates(df_hist_valid, valid_cutoff)

    # Merge features for train
    def assemble(cand, item_feats, user_feats, ui_feats, cutoff):
        df = cand.merge(item_feats, on="article_id", how="left")
        df = df.merge(user_feats, on="customer_id", how="left")
        df = df.merge(ui_feats, on=["customer_id", "article_id"], how="left")
        # recency fill
        for col in ["item_recency_days", "user_recency_days", "ui_recency_days"]:
            if col in df.columns:
                df[col] = df[col].fillna(9999).astype("float32")
        # counts/price fill
        count_cols = [c for c in df.columns if c.endswith(("7d", "14d", "28d", "84d")) or c.endswith("_cnt")]
        for col in count_cols:
            df[col] = df[col].fillna(0).astype("float32")
        if "ui_cnt" in df.columns:
            df["ui_cnt"] = df["ui_cnt"].fillna(0).astype("float32")

        # Merge article categorical metadata
        meta_cols = [
            "article_id",
            "product_type_no",
            "product_group_name",
            "index_code",
            "section_no",
            "graphical_appearance_no",
            "colour_group_code",
            "perceived_colour_value_id",
            "perceived_colour_master_id",
        ]
        df = df.merge(articles[meta_cols], on="article_id", how="left")

        # Cast selected to category dtype
        cat_cols = ["product_group_name", "index_code", "product_type_no", "section_no"]
        for c in cat_cols:
            if c in df.columns:
                # If numeric, convert to int then category to avoid excessive categories from NaN
                if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_float_dtype(df[c]):
                    df[c] = df[c].fillna(-1).astype("int32").astype("category")
                else:
                    df[c] = df[c].astype("category")

        # Keep feature set
        feature_cols = [
            # labels / ids
            # counts item
            "article_id_cnt_7d",
            "article_id_cnt_14d",
            "article_id_cnt_28d",
            "article_id_cnt_84d",
            "item_recency_days",
            "item_avg_price_84d",
            # counts user
            "customer_id_cnt_7d",
            "customer_id_cnt_14d",
            "customer_id_cnt_28d",
            "customer_id_cnt_84d",
            "user_recency_days",
            "user_avg_price_84d",
            # ui
            "ui_cnt",
            "ui_recency_days",
            # article meta
            "product_type_no",
            "product_group_name",
            "index_code",
            "section_no",
            "graphical_appearance_no",
            "colour_group_code",
            "perceived_colour_value_id",
            "perceived_colour_master_id",
        ]
        keep_cols = ["customer_id", "article_id", "label"] + feature_cols
        # Some columns may be missing (e.g., if no interactions); ensure they exist
        for c in feature_cols:
            if c not in df.columns:
                df[c] = 0
        df = df[keep_cols]
        return df

    print("Assembling feature matrices...")
    train_df = assemble(cand_train, item_train, user_train, ui_train, train_cutoff)
    valid_df = assemble(cand_valid, item_valid, user_valid, ui_valid, valid_cutoff)

    # Build groups for ranking
    def to_matrix(df):
        # Keep per-user grouping
        groups = df.groupby("customer_id").size().values.tolist()
        # Separate y and X
        y = df["label"].astype(int).values
        X = df.drop(columns=["label", "customer_id", "article_id"])
        return X, y, groups

    X_train, y_train, g_train = to_matrix(train_df)
    X_valid, y_valid, g_valid = to_matrix(valid_df)

    # XGBoost Ranker
    print("Training XGBRanker...")
    ranker = XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@12",
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=32,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=42,
        n_jobs=args.threads,
    )

    # Fit with groups
    ranker.fit(
        X_train,
        y_train,
        group=g_train,
        eval_set=[(X_valid, y_valid)],
        eval_group=[g_valid],
        verbose=True,
    )

    # Validate
    print("Scoring validation...")
    valid_pred = valid_df[["customer_id", "article_id", "label"]].copy()
    valid_pred["pred"] = ranker.predict(X_valid)
    map12 = evaluate_map12(valid_pred)
    print(f"Validation MAP@12: {map12:.5f}")

    # Save artifacts
    model_path = out_dir / "xgboost_ranker.json"
    ranker.save_model(model_path.as_posix())
    valid_pred_path = out_dir / "valid_predictions.parquet"
    valid_pred.to_parquet(valid_pred_path, index=False)
    print(f"Saved model to: {model_path}")
    print(f"Saved validation predictions to: {valid_pred_path}")


if __name__ == "__main__":
    main()