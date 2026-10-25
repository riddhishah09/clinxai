"""Stage 8: logic behind the web app (educational use only)."""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import gradio as gr
from src.model import load_split_data
from src.preprocessing import IMPOSSIBLE_ZERO_COLS
from src.lime_explainer import build_lime_explainer, explain_patient
from src.shap_explainer import build_shap_explainer, compute_shap, make_explanation

DISCLAIMER = ("**EDUCATIONAL PROJECT ONLY. This is not a medical device and must not be used "
              "for diagnosis or treatment decisions. See a qualified doctor for health concerns.**")

# (min, max) sanity limits for inputs
LIMITS = {"Pregnancies": (0, 20), "Glucose": (30, 300), "BloodPressure": (20, 200),
          "SkinThickness": (1, 120), "Insulin": (1, 1000), "BMI": (10, 80),
          "DiabetesPedigreeFunction": (0.05, 3.0), "Age": (18, 100)}
REQUIRED = ["Pregnancies", "DiabetesPedigreeFunction", "Age"]

_BUNDLE = {}


def _load():
    """Load model, imputer and explainers once."""
    if not _BUNDLE:
        X_train, _, _, _ = load_split_data()
        model = joblib.load("models/rf_model.joblib")
        _BUNDLE.update(
            features=list(X_train.columns),
            model=model,
            imputer=joblib.load("models/imputer.joblib"),
            lime=build_lime_explainer(X_train),
            shap=build_shap_explainer(model),
        )
    return _BUNDLE


def _clean_inputs(values, features):
    """Validate inputs; blanks (and impossible zeros) become NaN."""
    raw = {}
    for name, v in zip(features, values):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            if name in REQUIRED:
                raise gr.Error(f"Please enter a value for {name}.")
            v = np.nan
        elif name in IMPOSSIBLE_ZERO_COLS and v == 0:
            v = np.nan
        if not np.isnan(v):
            lo, hi = LIMITS[name]
            if not (lo <= v <= hi):
                raise gr.Error(f"{name} should be between {lo} and {hi}.")
        raw[name] = float(v)
    return pd.DataFrame([raw], columns=features)


def analyze_patient(preg, glucose, bp, skin, insulin, bmi, dpf, age):
    """Return: summary text, input table, LIME figure, SHAP waterfall, SHAP force."""
    b = _load()
    feats = b["features"]
    plt.close("all")

    # NOTE: order of the inputs must match the feature order from Stage 1
    raw_df = _clean_inputs([preg, glucose, bp, skin, insulin, bmi, dpf, age], feats)
    was_missing = raw_df.iloc[0].isna()
    row_df = pd.DataFrame(b["imputer"].transform(raw_df), columns=feats)
    row = row_df.iloc[0]

    prob = float(b["model"].predict_proba(row_df)[0, 1])
    label = "Diabetes (positive)" if prob > 0.5 else "No diabetes (negative)"
    summary = (f"### Predicted risk of diabetes: {prob:.1%}\n"
               f"Model output at the 0.5 threshold: **{label}**\n\n"
               f"{DISCLAIMER}")

    table = pd.DataFrame({
        "Feature": feats,
        "Value used": row.round(2).values,
        "Filled with training median?": ["Yes" if m else "No" for m in was_missing.values],
    })

    # LIME
    lime_exp = explain_patient(b["lime"], b["model"], row, num_samples=3000)
    lime_fig = lime_exp.as_pyplot_figure()
    lime_fig.set_size_inches(8, 4.5)
    lime_fig.tight_layout()

    # SHAP waterfall
    values, base = compute_shap(b["shap"], row_df)
    shap.plots.waterfall(make_explanation(values[0], base, row), max_display=8, show=False)
    wf_fig = plt.gcf()
    wf_fig.tight_layout()

    # SHAP force plot (matplotlib version)
    force_fig = None
    try:
        shap.force_plot(base, values[0], row.round(2).values, feature_names=feats,
                        matplotlib=True, show=False, figsize=(14, 3.5))
        force_fig = plt.gcf()
    except Exception as e:
        print("Force plot skipped:", type(e).__name__, e)

    return summary, table, lime_fig, wf_fig, force_fig
