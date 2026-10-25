"""Stage 4: Random Forest training and evaluation (educational use only)."""
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

RANDOM_STATE = 42


def load_split_data():
    """Load the imputed train/test files saved in Stage 3."""
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze("columns")
    y_test = pd.read_csv("data/y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    """Train a Random Forest with sensible beginner-friendly settings."""
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y):
    """Return a dictionary of metrics for the given data."""
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def save_metrics(metrics, path="outputs/metrics.json"):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
