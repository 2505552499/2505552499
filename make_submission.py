import argparse
import numpy as np
import pandas as pd
import os


def load_weights(path, default):
    if os.path.exists(path):
        w = np.load(path)
        if w.ndim == 0:
            w = np.array([float(w)])
        w = w / w.sum()
        return w
    return np.array(default)


def main(args):
    df = pd.read_parquet(args.test_file)
    prob_m0 = np.load(args.prob_m0)
    prob_m3 = np.load(args.prob_m3)[:, [1, 0, 2]]
    prob_qlora = np.load(args.prob_qlora)

    probs = [prob_m0, prob_m3, prob_qlora]
    weights = load_weights(args.weights, [2.0, 0.99, 0.0])
    ensemble = sum(w * p for w, p in zip(weights, probs))

    sub = pd.DataFrame({
        "id": df["id"],
        "winner_model_a": ensemble[:, 0],
        "winner_model_b": ensemble[:, 1],
        "winner_tie": ensemble[:, 2],
    })
    sub.to_csv(args.output, index=False)
    print(sub.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create submission from saved probabilities")
    parser.add_argument("--test_file", default="test.parquet")
    parser.add_argument("--prob_m0", default="prob_m0.npy")
    parser.add_argument("--prob_m3", default="prob_m3.npy")
    parser.add_argument("--prob_qlora", default="prob_qlora.npy")
    parser.add_argument("--weights", default="opt_weights.npy", help="Path to ensemble weights")
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()
    main(args)
