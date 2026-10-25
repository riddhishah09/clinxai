"""Stage 5: LIME explanations (educational use only)."""
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

CLASS_NAMES = ["No diabetes", "Diabetes"]


def build_lime_explainer(X_train, random_state=42):
    """LIME learns feature ranges from the TRAINING data to create perturbations."""
    return LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=list(X_train.columns),
        class_names=CLASS_NAMES,
        mode="classification",
        discretize_continuous=True,
        random_state=random_state,
    )


def make_predict_fn(model, feature_names):
    """Wrap predict_proba so LIME's numpy arrays become DataFrames (keeps feature names)."""
    def predict_fn(arr):
        return model.predict_proba(pd.DataFrame(arr, columns=feature_names))
    return predict_fn


def explain_patient(explainer, model, row, num_features=8, num_samples=5000):
    """Explain one patient (a pandas Series). Explains the 'Diabetes' class (label 1)."""
    predict_fn = make_predict_fn(model, list(row.index))
    return explainer.explain_instance(
        row.values, predict_fn, num_features=num_features, num_samples=num_samples
    )


def surrogate_r2(exp):
    """R^2 of LIME's local linear model (how well it mimics the forest nearby)."""
    score = exp.score
    if isinstance(score, dict):
        score = score[1]
    return float(np.ravel(score)[0])


def select_cases(model, X_test, y_test):
    """Pick representative test patients automatically from real predictions."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba > 0.5).astype(int)
    y = np.asarray(y_test)

    def pick(mask, func):
        idx = np.where(mask)[0]
        return None if len(idx) == 0 else int(idx[func(proba[idx])])

    cases = {
        "true_positive": pick((y == 1) & (pred == 1), np.argmax),
        "true_negative": pick((y == 0) & (pred == 0), np.argmin),
        "false_negative": pick((y == 1) & (pred == 0), np.argmin),
        "false_positive": pick((y == 0) & (pred == 1), np.argmax),
        "borderline": int(np.argmin(np.abs(proba - 0.5))),
    }
    return {k: v for k, v in cases.items() if v is not None}


def lime_stability(model, X_train, row, seeds=(0, 1, 2, 3, 4), top_k=3):
    """Re-run LIME with different seeds; return the top-k features of each run."""
    results = []
    for s in seeds:
        explainer = build_lime_explainer(X_train, random_state=s)
        exp = explain_patient(explainer, model, row)
        top = sorted(exp.as_map()[1], key=lambda t: -abs(t[1]))[:top_k]
        results.append([X_train.columns[i] for i, _ in top])
    return results
