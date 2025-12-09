import os
import gc
import json
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import optuna
import lightgbm as lgb
from functools import partial



# =============================================================
# Paths
# =============================================================
META_PATH = "../data/outputs/dataset_meta.json"
TRAIN_RANK_PATH = "../data/outputs/train_rank.parquet"
VALID_RANK_PATH = "../data/outputs/valid_rank.parquet"
GROUP_TRAIN_PATH = "../data/outputs/groups_train.npy"
GROUP_VALID_PATH = "../data/outputs/groups_valid.npy"

OUTPUT_DIR = "../data/outputs/optuna_lgbm_ranker/"
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
    """Keep only groups with ≥1 positive."""
    print("Filtering validation groups with ≥1 positive...")

    group_bounds = np.insert(np.cumsum(group_valid), 0, 0)[:-1]
    has_pos = np.add.reduceat(df_valid["label"].values, group_bounds) > 0

    new_groups = group_valid[has_pos]
    row_mask = np.repeat(has_pos, group_valid)

    df_valid = df_valid[row_mask].reset_index(drop=True)

    print(f" → Groups: {len(group_valid)} → {len(new_groups)}")
    print(f" → Rows:   {len(df_valid)}")

    return df_valid, new_groups


# =============================================================
# Optuna Objective  (UPDATED: correct depth + num_leaves logic)
# =============================================================
def objective(trial, train_df, train_group, valid_df, valid_group, feature_cols):

    max_depth = trial.suggest_int("max_depth", 5, 12)

    # setting leaves to 2**max depth but with a ceiling of 1024 leaves
    num_leaves = min(2 ** max_depth, 1024)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [12],
        "verbosity": -1,
        "learning_rate": trial.suggest_float("eta", 0.02, 0.15, log=True),
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "feature_fraction": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("subsample", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 5),
        "min_data_in_leaf": trial.suggest_int("min_child_weight", 1, 200),
        "min_sum_hessian_in_leaf": trial.suggest_float(
            "min_sum_hessian_in_leaf", 1e-3, 50.0, log=True
        ),
        "lambda_l1": trial.suggest_float("alpha", 0.0, 10.0),
        "lambda_l2": trial.suggest_float("lambda", 1.0, 40.0),
        "max_bin": trial.suggest_int("max_bin", 128, 511),
        "n_estimators": 500,
    }

    model = lgb.LGBMRanker(**params)

    model.fit(
        train_df[feature_cols],
        train_df["label"],
        group=train_group,
        eval_set=[(valid_df[feature_cols], valid_df["label"])],
        eval_group=[valid_group],
        eval_names=["valid"],
        eval_at=[12],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    return model.best_score_["valid"]["ndcg@12"]


# =============================================================
# Main routine with group-based sampling
# =============================================================
def run_optuna(n_trials=100, valid_fraction=0.3):

    # -------------------------------------------------------------
    # Load metadata & features
    # -------------------------------------------------------------
    print("Loading metadata...")
    with open(META_PATH) as f:
        meta = json.load(f)

    feature_cols = meta["model_features"]

    # -------------------------------------------------------------
    # Load train
    # -------------------------------------------------------------
    print("Loading train...")
    train_df = pd.read_parquet(TRAIN_RANK_PATH)
    train_group = np.load(GROUP_TRAIN_PATH)

    for c in feature_cols:
        if train_df[c].dtype == "float64":
            train_df[c] = train_df[c].astype("float32")

    # -------------------------------------------------------------
    # Load validation metadata
    # -------------------------------------------------------------
    print("Loading validation group sizes...")
    group_full = np.load(GROUP_VALID_PATH)
    n_groups = len(group_full)
    bounds = np.insert(np.cumsum(group_full), 0, 0)

    # -------------------------------------------------------------
    # Sample groups
    # -------------------------------------------------------------
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
        row_mask[bounds[g]:bounds[g+1]] = True

    # -------------------------------------------------------------
    # Load validation (STREAMING, WITH FIXED OFFSET)
    # -------------------------------------------------------------
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

        for c in feature_cols:
            if df_chunk[c].dtype == "float64":
                df_chunk[c] = df_chunk[c].astype("float32")

        valid_parts.append(df_chunk)

        offset += chunk_len
        del df_chunk
        gc.collect()

    valid_df = pd.concat(valid_parts, ignore_index=True)
    del valid_parts
    gc.collect()

    valid_group = np.array([group_full[g] for g in selected_groups], dtype=np.int32)

    print("Valid rows before filtering:", len(valid_df))

    # -------------------------------------------------------------
    # Filter groups with ≥1 positive
    # -------------------------------------------------------------
    valid_df, valid_group = filter_validation(valid_df, valid_group)

    # -------------------------------------------------------------
    # Optuna study
    # -------------------------------------------------------------
    study = optuna.create_study(
        direction="maximize",
        study_name="lgbm_ranker_opt",
        storage=f"sqlite:///{OUTPUT_DIR}/lgbm_optuna.db",
        load_if_exists=True,
    )

    obj = partial(
        objective,
        train_df=train_df,
        train_group=train_group.tolist(),
        valid_df=valid_df,
        valid_group=valid_group.tolist(),
        feature_cols=feature_cols,
    )

    study.optimize(obj, n_trials=n_trials, callbacks=[trial_progress_callback])

    print("\n==============================")
    print("Best Params:", study.best_params)
    print("Best Valid NDCG:", study.best_value)
    print("==============================\n")

    # -------------------------------------------------------------
    # Final full training
    # -------------------------------------------------------------
    print("Training final full model...")

    best_params = study.best_params.copy()
    best_params.update({
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [12],
        "verbosity": -1,
        "n_estimators": 3000,
    })

    model = lgb.LGBMRanker(**best_params)

    evals_result = {}

    model.fit(
        train_df[feature_cols],
        train_df["label"],
        group=train_group,
        eval_set=[(train_df[feature_cols], train_df["label"]),
                  (valid_df[feature_cols], valid_df["label"])],
        eval_group=[train_group, valid_group],
        eval_names=["train", "valid"],
        eval_at=[12],
        callbacks=[
            lgb.early_stopping(200, verbose=True),
            lgb.record_evaluation(evals_result),
        ],
    )

    # Save model
    model_path = os.path.join(OUTPUT_DIR, "best_lgbm_ranker.model")
    model.booster_.save_model(model_path)
    print(f"Saved final model to {model_path}")

    # -------------------------------------------------------------
    # Save Training Log
    # -------------------------------------------------------------
    log_path = os.path.join(OUTPUT_DIR, "lgbm_final_training_log.txt")

    train_log = evals_result["train"]["ndcg@12"]
    valid_log = evals_result["valid"]["ndcg@12"]

    max_len = min(len(train_log), len(valid_log))

    with open(log_path, "w") as f:
        f.write("iter,train_ndcg12,valid_ndcg12\n")
        for i in range(max_len):
            f.write(f"{i},{train_log[i]},{valid_log[i]}\n")

    print(f"Saved training log to: {log_path}")



# =============================================================
# CLI
# =============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--valid_fraction", type=float, default=0.3)
    args = parser.parse_args()

    run_optuna(n_trials=args.trials, valid_fraction=args.valid_fraction)
