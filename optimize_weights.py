import argparse
import numpy as np
from sklearn.metrics import log_loss
from scipy.optimize import minimize


def load_probs(paths):
    return [np.load(p) for p in paths]


def objective(weights, probs, y):
    weights = np.clip(weights, 0, None)
    weights = weights / weights.sum()
    ensemble = sum(w * p for w, p in zip(weights, probs))
    return log_loss(y, ensemble)


def optimize(probs, y):
    init = np.ones(len(probs))
    result = minimize(objective, init, args=(probs, y), method="Powell")
    weights = np.clip(result.x, 0, None)
    weights = weights / weights.sum()
    return weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune ensemble weights using validation labels")
    parser.add_argument("labels", help="Path to numpy array with one-hot validation labels")
    parser.add_argument("prob_files", nargs="+", help="List of probability prediction files")
    parser.add_argument("--output", default="opt_weights.npy", help="Output file for weights")
    args = parser.parse_args()

    probs = load_probs(args.prob_files)
    y = np.load(args.labels)
    weights = optimize(probs, y)
    np.save(args.output, weights)
    print("Optimal weights", weights)
