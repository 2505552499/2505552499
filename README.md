# Kaggle LLM Classification Finetuning

This repository contains a notebook for LLM classification fine-tuning and a couple of helper scripts.

## New scripts

- `optimize_weights.py` – tunes ensemble weights for multiple model predictions using a validation dataset. It expects probability arrays (`.npy`) from different models and the ground truth labels in one-hot format. The script saves the optimal weights to `opt_weights.npy`.

- `make_submission.py` – generates `submission.csv` by combining model probabilities with the specified weights. If `opt_weights.npy` is present it will be used, otherwise default weights `[2.0, 0.99, 0.0]` are applied.

Both scripts rely on numpy, pandas, scikit-learn and scipy. They are designed to run in the Kaggle environment where the necessary prediction files are available.

```bash
# Example usage
python optimize_weights.py y_val.npy prob_m0.npy prob_m3.npy prob_qlora.npy
python make_submission.py --test_file test.parquet
```
