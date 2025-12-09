import os
import gc
import json
import argparse
from functools import partial

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import optuna
from catboost import CatBoostRanker, Pool

BATCH = 2_000_000

# =============================================================
# Paths
# =============================================================
META_PATH = "../data/outputs/dataset_meta.json"
TRAIN_RANK_PATH = "../data/outputs/train_rank.parquet"
VALID_RANK_PATH = "../data/outputs/valid_rank.parquet"
GROUP_TRAIN_PATH = "../data/outputs/groups_train.npy"
GROUP_VALID_PATH = "../data/outputs/groups_valid.npy"

OUTPUT_DIR = "../data/outputs/optuna_catboost/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================
# Utilities
# =============================================================
def trial_progress_callback(study, trial):
    print(
        f"Trial {trial.number} | status={trial.state} | "
        f"value={trial.value:.6f} | best={study.best_value:.6f}"
    )


def filter_validation(df_valid, group_valid):
    print("Filtering validation groups with ≥1 positive...")

    group_bounds = np.insert(np.cumsum(group_valid), 0, 0)[:-1]
    has_pos = np.add.reduceat(df_valid['label'].values, group_bounds) > 0

    filtered_group = group_valid[has_pos]
    row_mask = np.repeat(has_pos, group_valid)

    df_valid = df_valid[row_mask].reset_index(drop=True)

    print(f" → Groups: {len(group_valid)} → {len(filtered_group)}")
    print(f" → Rows:   {len(df_valid)}")

    return df_valid, filtered_group


def expand_group_ids(group_sizes):
    """
    Convert group sizes (length = n_groups) into group_id per row
    for CatBoost: group_id[i] = group index of row i.
    """
    return np.repeat(np.arange(len(group_sizes), dtype=np.int32), group_sizes)


def find_ndcg_key(eval_dict):
    """
    Find the key for NDCG metric inside CatBoost evals_result dict.
    Falls back to first metric if NDCG not explicitly found.
    """
    for key in eval_dict:
        if "NDCG" in key.upper():
            return key
    available = list(eval_dict.keys())
    if available:
        print(
            f"Warning: NDCG not found in eval dict. "
            f"Using first metric: {available[0]}"
        )
        return available[0]
    raise KeyError(f"No metrics found in eval_dict. Keys: {available}")


# =============================================================
# Optuna Objective
# =============================================================
def objective(trial, pool_train, pool_valid):
    """
    Optuna objective for CatBoostRanker.
    We maximize NDCG@12 on the validation set.
    """

    params = {
        "loss_function": "YetiRank",
        "eval_metric": "NDCG:top=12",
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "depth": trial.suggest_int("depth", 5, 12),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 40.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 200),
        "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 64, 512),
        "task_type": "GPU",
        "devices": "0",
        "random_seed": 42,
        "iterations": 5000,
        "use_best_model": True,
        "od_type": "Iter",
        "od_wait": 200,
        "verbose": False,
    }

    model = CatBoostRanker(**params)
    model.fit(pool_train, eval_set=pool_valid)

    evals = model.get_evals_result()
    valid_metrics = evals.get("validation", {})

    metric_key = find_ndcg_key(valid_metrics)
    valid_log = valid_metrics[metric_key]

    best_score = max(valid_log)
    return best_score


# =============================================================
# Main routine with group-based sampling
# =============================================================
def run_optuna(n_trials=100, valid_fraction=0.3):

    # ---------------------------------------------------------
    # Load metadata & feature list
    # ---------------------------------------------------------
    print("Loading metadata...")
    with open(META_PATH) as f:
        meta = json.load(f)

    feature_cols = meta["model_features"]

    # ---------------------------------------------------------
    # Load train
    # ---------------------------------------------------------
    print("Loading train...")
    train_df = pd.read_parquet(TRAIN_RANK_PATH)
    train_group = np.load(GROUP_TRAIN_PATH)

    # Cast floats to float32
    for c in feature_cols:
        if train_df[c].dtype == "float64":
            train_df[c] = train_df[c].astype("float32")

    train_df["label"] = train_df["label"].astype("float32")

    # ---------------------------------------------------------
    # Load validation group sizes
    # ---------------------------------------------------------
    print("Loading validation group sizes...")
    group_full = np.load(GROUP_VALID_PATH)
    n_groups = len(group_full)
    bounds = np.insert(np.cumsum(group_full), 0, 0)

    # ---------------------------------------------------------
    # Group-based sampling for validation
    # ---------------------------------------------------------
    if valid_fraction < 1.0:
        rng = np.random.default_rng(42)
        n_keep = int(n_groups * valid_fraction)
        selected_groups = sorted(rng.choice(n_groups, n_keep, replace=False))
        print(f"Using {n_keep}/{n_groups} validation groups (~{valid_fraction*100:.0f}%)")
    else:
        selected_groups = list(range(n_groups))
        print("Using 100% of validation groups")

    total_rows = bounds[-1]
    row_mask = np.zeros(total_rows, dtype=bool)

    for g in selected_groups:
        row_mask[bounds[g]:bounds[g + 1]] = True

   # ---------------------------------------------------------
    # Load VALID in batches (to avoid OOM)
    # ---------------------------------------------------------
    print(f"Loading VALID in batches (batch_size={BATCH})...")
    pf = pq.ParquetFile(VALID_RANK_PATH)
    valid_parts = []
    pos = 0

    for batch in pf.iter_batches(batch_size=BATCH):
        batch_len = batch.num_rows
        batch_mask = row_mask[pos : pos + batch_len]
        if batch_mask.any():
            pdf = batch.to_pandas()
            pdf = pdf.loc[batch_mask].reset_index(drop=True)

            for c in feature_cols:
                if pdf[c].dtype == "float64":
                    pdf[c] = pdf[c].astype("float32")
            if "label" in pdf.columns:
                pdf["label"] = pdf["label"].astype("float32")

            valid_parts.append(pdf)

        pos += batch_len

    valid_df = pd.concat(valid_parts, ignore_index=True)
    del valid_parts
    gc.collect()

    valid_group = np.array([group_full[g] for g in selected_groups], dtype=np.int32)

    print("Valid rows before filtering:", len(valid_df))


    # ---------------------------------------------------------
    # Filter validation groups with ≥1 positive label
    # ---------------------------------------------------------
    valid_df, valid_group = filter_validation(valid_df, valid_group)


    # ---------------------------------------------------------
    # Build CatBoost Pools
    # ---------------------------------------------------------
    print("Building CatBoost Pools...")

    train_group_id = expand_group_ids(train_group)
    valid_group_id = expand_group_ids(valid_group)

    assert len(train_group_id) == len(train_df)
    assert len(valid_group_id) == len(valid_df)

    cat_features = []

    pool_train = Pool(
        data=train_df[feature_cols],
        label=train_df["label"],
        group_id=train_group_id,
        cat_features=cat_features,
    )

    pool_valid = Pool(
        data=valid_df[feature_cols],
        label=valid_df["label"],
        group_id=valid_group_id,
        cat_features=cat_features,
    )

    print("Train rows:", pool_train.shape[0])
    print("Valid rows:", pool_valid.shape[0])

   # ---------------------------------------------------------
    # Optuna Study
    # ---------------------------------------------------------
    study = optuna.create_study(
        direction="maximize",
        study_name="catboost_ranker_opt",
        storage=f"sqlite:///{OUTPUT_DIR}/catboost_optuna.db",
        load_if_exists=True,
    )

    obj = partial(objective, pool_train=pool_train, pool_valid=pool_valid)

    study.optimize(obj, n_trials=n_trials, n_jobs=1, callbacks=[trial_progress_callback])


    print("\n==============================")
    print("Best Params:", study.best_params)
    print("Best Valid NDCG:", study.best_value)
    print("==============================\n")

    # ---------------------------------------------------------
    # Final full training with more iterations + logging
    # ---------------------------------------------------------
    print("Training final full CatBoost model...")

    best_params = study.best_params.copy()
    best_params.update(
        {
            "loss_function": "YetiRank",
            "eval_metric": "NDCG:top=12",
            "task_type": "GPU",
            "devices": "0",
            "random_seed": 42,
            "iterations": 5000,
            "use_best_model": True,
            "od_type": "Iter",
            "od_wait": 200,
            "verbose": 50,
            "thread_count": -1,
        }
    )
    final_model = CatBoostRanker(**best_params)
    final_model.fit(pool_train, eval_set=pool_valid)

    best_iter = final_model.get_best_iteration()
    print("Best iteration:", best_iter)

    # Shrink to best iteration
    if best_iter is not None and best_iter > 0:
        final_model.shrink(ntree_end=best_iter + 1)

    # Save model
    model_path = os.path.join(OUTPUT_DIR, "best_catboost_ranker.model")
    final_model.save_model(model_path)
    print(f"Saved final model to {model_path}")

    # ---------------------------------------------------------
    # Save Training Log (similar to your CatBoost script)
    # ---------------------------------------------------------
    log_path = os.path.join(OUTPUT_DIR, "catboost_final_training_log.txt")

    evals = final_model.get_evals_result()
    valid_metrics = evals.get("validation", {})

    valid_metric_key = find_ndcg_key(valid_metrics)
    valid_log = valid_metrics[valid_metric_key]

    with open(log_path, "w") as f:
        f.write("iter,valid_ndcg12\n")
        for i, val in enumerate(valid_log):
            f.write(f"{i},{val}\n")

    print(f"Saved training log to: {log_path}")

    # Cleanup
    del pool_train, pool_valid, final_model
    gc.collect()


# =============================================================
# CLI
# =============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--valid_fraction", type=float, default=0.3)
    args = parser.parse_args()

    run_optuna(n_trials=args.trials, valid_fraction=args.valid_fraction)