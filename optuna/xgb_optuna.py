import os
import gc
import json
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import optuna
import xgboost as xgb
from optuna.integration import XGBoostPruningCallback
from functools import partial

def trial_progress_callback(study, trial):
    print(
        f"Trial {trial.number} finished "
        f"status={trial.state}, "
        f"value={trial.value:.6f}, "
        f"best={study.best_value:.6f}"
    )


# =============================================================
# Paths
# =============================================================
META_PATH = "../data/outputs/dataset_meta.json"
TRAIN_RANK_PATH = "../data/outputs/train_rank.parquet"
VALID_RANK_PATH = "../data/outputs/valid_rank.parquet"
GROUP_TRAIN_PATH = "../data/outputs/groups_train.npy"
GROUP_VALID_PATH = "../data/outputs/groups_valid.npy"

OUTPUT_DIR = "../data/outputs/optuna_xgb/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================
# Filter validation groups with ≥1 positive
# =============================================================
def filter_validation(df_valid, group_valid):
    print("Filtering validation groups with ≥1 positive...")

    group_bounds = np.insert(np.cumsum(group_valid), 0, 0)[:-1]
    group_has_pos = np.add.reduceat(df_valid["label"].values, group_bounds) > 0

    filtered_group = group_valid[group_has_pos]
    row_mask = np.repeat(group_has_pos, group_valid)

    df_valid = df_valid[row_mask].reset_index(drop=True)

    print(f" → Groups: {len(group_valid)} → {len(filtered_group)}")
    print(f" → Rows:   {len(df_valid)}")

    return df_valid, filtered_group


# =============================================================
# Build DMatrix
# =============================================================
def build_dmatrix(df, feature_cols, group_sizes):
    d = xgb.DMatrix(
        data=df[feature_cols],
        label=df["label"].astype(np.float32).values
    )
    d.set_group(group_sizes)
    return d


# =============================================================
# Optuna Objective
# =============================================================
def objective(trial, train_df, train_group, valid_df, valid_group, feature_cols):

    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@12",
        "tree_method": "hist",      # CPU only
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "eta": trial.suggest_float("eta", 0.02, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 200.0),
        "lambda": trial.suggest_float("lambda", 1.0, 40.0),
        "alpha": trial.suggest_float("alpha", 0.0, 10.0),
        "nthread": -1,
        "verbosity": 0,
    }

    dtrain = build_dmatrix(train_df, feature_cols, train_group)
    dvalid = build_dmatrix(valid_df, feature_cols, valid_group)

    pruning_callback = XGBoostPruningCallback(
        trial, "valid-ndcg@12"
    )

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        num_boost_round=500,
        callbacks=[pruning_callback],
        verbose_eval=False,
    )

    metrics = booster.eval(dvalid)
    ndcg = float(metrics.split(":")[-1])

    return -ndcg


# =============================================================
# Main optuna routine with GROUP-BASED sampling
# =============================================================
def run_optuna(n_trials=30, valid_fraction=1.0):
    print("Loading metadata...")
    with open(META_PATH) as f:
        meta = json.load(f)
    feature_cols = meta["model_features"]

    # =============================================================
    # Load Train
    # =============================================================
    print("Loading train...")
    train_df = pd.read_parquet(TRAIN_RANK_PATH)
    train_group = np.load(GROUP_TRAIN_PATH)

    for c in feature_cols:
        if train_df[c].dtype == "float64":
            train_df[c] = train_df[c].astype("float32")

    # =============================================================
    # Load Validation Groups for Fraction Sampling
    # =============================================================
    print("Loading validation group sizes...")
    valid_group_full = np.load(GROUP_VALID_PATH)
    n_groups = len(valid_group_full)

    bounds = np.insert(np.cumsum(valid_group_full), 0, 0)

    # -------------------------------------------------------------
    # Sample groups (not rows)
    # -------------------------------------------------------------
    if valid_fraction < 1.0:
        rng = np.random.default_rng(42)
        n_keep = int(n_groups * valid_fraction)
        selected_groups = sorted(rng.choice(n_groups, n_keep, replace=False))
        print(f"Keeping {n_keep} of {n_groups} validation groups")
    else:
        selected_groups = list(range(n_groups))
        print("Using 100% of validation groups")

    # Build a row_mask for all selected groups
    total_rows = bounds[-1]
    row_mask = np.zeros(total_rows, dtype=bool)

    for g in selected_groups:
        start = bounds[g]
        end = bounds[g + 1]
        row_mask[start:end] = True

    # =============================================================
    # Load VALID in batches and apply row_mask
    # =============================================================
    print("Loading VALID (stream)...")
    pf = pq.ParquetFile(VALID_RANK_PATH)
    valid_parts = []

    offset = 0
    BATCH = 5_000_000

    for batch in pf.iter_batches(batch_size=BATCH):
        df_chunk = batch.to_pandas()
        chunk_len = len(df_chunk)

        mask_chunk = row_mask[offset:offset + chunk_len]
        df_chunk = df_chunk[mask_chunk]

        valid_parts.append(df_chunk)

        offset += chunk_len
        del df_chunk; gc.collect()

    valid_df = pd.concat(valid_parts, ignore_index=True)
    del valid_parts; gc.collect()

    # Build filtered group sizes list
    valid_group = np.array([valid_group_full[g] for g in selected_groups], dtype=np.int32)

    print("Valid rows before filtering:", len(valid_df))

    # =============================================================
    # Filter groups with ≥1 positive
    # =============================================================
    valid_df, valid_group = filter_validation(valid_df, valid_group)

    # =============================================================
    # Optuna Setup
    # =============================================================
    study_path = os.path.join(OUTPUT_DIR, "xgb_optuna.db")

    study = optuna.create_study(
        direction="minimize",
        study_name="xgb_ranker_opt",
        storage=f"sqlite:///{study_path}",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=20,
            interval_steps=10
        ),
    )

    obj = partial(
        objective,
        train_df=train_df,
        train_group=train_group,
        valid_df=valid_df,
        valid_group=valid_group,
        feature_cols=feature_cols,
    )

    study.optimize(obj, n_trials=n_trials, callbacks=[trial_progress_callback])

    print("\n==============================")
    print("Best Params:", study.best_params)
    print("Best Valid NDCG:", -study.best_value)
    print("==============================")


    # =============================================================
    # Final Full Training with Early Stopping
    # =============================================================
    print("Training final full model with early stopping...")

    best_params = study.best_params
    best_params.update({
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@12",
        "tree_method": "hist",
        "nthread": -1,
    })

    dtrain = build_dmatrix(train_df, feature_cols, train_group)
    dvalid = build_dmatrix(valid_df, feature_cols, valid_group)

    booster = xgb.train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=3000,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=200,
        verbose_eval=50
    )

    model_path = os.path.join(OUTPUT_DIR, "best_xgb_ranker.model")
    booster.save_model(model_path)

    print(f"Saved final model to {model_path}")


# =============================================================
# CLI
# =============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--valid_fraction", type=float, default=1.0)
    args = parser.parse_args()

    run_optuna(n_trials=args.trials, valid_fraction=args.valid_fraction)
