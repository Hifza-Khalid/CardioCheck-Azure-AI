"""Train a from-scratch Perceptron on the UCI Cleveland heart dataset."""
import json
import numpy as np
import pandas as pd

url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "heart-disease/processed.cleveland.data")
cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal","num"]
df = pd.read_csv(url, names=cols, na_values="?")

FEATS = cols[:-1]
X = df[FEATS].apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.mean())
y = (df["num"] > 0).astype(int).values

mu, sd = X.mean().values, X.std().values
sd[sd == 0] = 1.0
Z = (X.values - mu) / sd

rng = np.random.default_rng(42)
idx = rng.permutation(len(Z))
split = int(0.8 * len(Z))
tr, te = idx[:split], idx[split:]
Z_tr, y_tr, Z_te, y_te = Z[tr], y[tr], Z[te], y[te]

w, b, lr = np.zeros(Z.shape[1]), 0.0, 0.01
for _ in range(200):
    errors = 0
    for xi, yi in zip(Z_tr, y_tr):
        pred = 1 if (xi @ w + b) > 0 else 0
        if pred != yi:
            w += lr * (yi - pred) * xi
            b += lr * (yi - pred)
            errors += 1
    if errors == 0:
        break

def metrics(Z, y):
    p = (Z @ w + b > 0).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"accuracy": round(acc,4), "precision": round(prec,4),
            "recall": round(rec,4), "f1": round(f1,4)}

model = {
    "features": FEATS,
    "mean": [round(float(v),6) for v in mu],
    "std": [round(float(v),6) for v in sd],
    "weights": [round(float(v),6) for v in w],
    "bias": round(float(b),6),
    "train": metrics(Z_tr, y_tr),
    "test": metrics(Z_te, y_te),
}
with open("model.json","w") as f:
    json.dump(model, f, indent=2)
print("Train:", model["train"]); print("Test: ", model["test"])