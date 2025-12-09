# Personalized Fashion Recommendations (H&M)

This repository contains a fully functional setup for the H&M Personalized Fashion Recommendation competition.
The workflow is not yet fully automated, so this guide explains the exact order in which to run each step.

Minimal run order for the competition setup. Keep the dates below in sync with the `date` field at the top of each notebook.

Rolling weeks used for steps 1–3:
- 2020-08-19
- 2020-08-26
- 2020-09-02
- 2020-09-09
- 2020-09-16
- 2020-09-22

### Run Order
1) **Generate candidates** — run each generator notebook once per date (order does not matter; can run in parallel):
   - `c_user_user.ipynb`
   - `c_same_product.ipynb`
   - `c_repurchase.ipynb`
   - `c_age_cluster.ipynb`
   - `c_recent_top.ipynb`
   - `c_itemcf.ipynb`
   - `c_weekly_trending.ipynb`
   - `c_embeddings.ipynb`
   - `c_popularity.ipynb`
   - `c_category_affinity.ipynb`

2) **Build candidates** — `processing/build_candidates.ipynb`, once per date, to assemble candidates for each rolling week.

3) **Attach features** — `processing/features.ipynb`, once per date, to add features to all candidates.

4) **Prepare model dataset** — `processing/prepare_model_dataset.ipynb` once after steps 1–3 are complete.

5) **(Optional) Tune** — run notebooks in `optuna/` to search hyperparameters, or reuse existing ones.

6) **Train rankers** — run `models/lightgbm_ranker.ipynb`, `models/catboost_ranker.ipynb`, and `models/xgboost_ranker.ipynb`.

7) **(Optional) Ensemble** — `models/ensamble_inference.ipynb` to blend model outputs.