"""Stage 6: SHAP explanations (educational use only)."""
import numpy as np
import pandas as pd
import shap


def build_shap_explainer(model):
    """TreeExplainer: exact Shapley values for tree ensembles."""
    return shap.TreeExplainer(model)


def compute_shap(explainer, X):
    """Return (values, base) for the 'Diabetes' class (label 1).
    values has shape (n_rows, n_features). Handles old and new shap formats."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):          # old shap: list of arrays, one per class
        values = np.asarray(sv[1])
    else:
        sv = np.asarray(sv)
        values = sv[:, :, 1] if sv.ndim == 3 else sv
    base = np.ravel(explainer.expected_value)
    base = float(base[1]) if len(base) > 1 else float(base[0])
    return values, base


def additivity_check(model, X, values, base):
    """Max difference between (base + sum SHAP) and the model's probability."""
    recon = base + values.sum(axis=1)
    proba = model.predict_proba(X)[:, 1]
    return float(np.max(np.abs(recon - proba)))


def make_explanation(values_row, base, row):
    """Build a shap.Explanation for one patient (used by waterfall plots)."""
    return shap.Explanation(
        values=values_row,
        base_values=base,
        data=row.values,
        feature_names=list(row.index),
    )


def global_importance(values, feature_names):
    """Mean absolute SHAP value per feature, sorted high to low."""
    return (pd.Series(np.abs(values).mean(axis=0), index=feature_names)
            .sort_values(ascending=False))
