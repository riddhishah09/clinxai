"""ClinXAI Streamlit web interface (educational use only).
Run from the project folder:  streamlit run app_streamlit.py
Needs: src/, models/rf_model.joblib, models/imputer.joblib, data/X_train.csv, data/X_test.csv
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.model import load_split_data
from src.lime_explainer import build_lime_explainer, explain_patient
from src.shap_explainer import (build_shap_explainer, compute_shap,
                                make_explanation, global_importance)

DISCLAIMER = ("EDUCATIONAL PROJECT ONLY. This is not a medical device and must not be used "
              "for diagnosis or treatment decisions. See a qualified doctor for health concerns.")

# name: (label, min, max, default, step, optional)
FIELDS = {
    "Pregnancies": ("Pregnancies", 0, 20, 2, 1, False),
    "Glucose": ("Glucose (mg/dL)", 30, 300, 120, 1, True),
    "BloodPressure": ("Blood pressure, diastolic (mm Hg)", 20, 200, 70, 1, True),
    "SkinThickness": ("Skin thickness (mm)", 1, 120, 25, 1, True),
    "Insulin": ("Insulin (mu U/ml)", 1, 1000, 100, 1, True),
    "BMI": ("BMI (kg/m2)", 10.0, 80.0, 28.0, 0.1, True),
    "DiabetesPedigreeFunction": ("Diabetes pedigree function", 0.05, 3.0, 0.4, 0.01, False),
    "Age": ("Age (years)", 18, 100, 35, 1, False),
}

st.set_page_config(page_title="ClinXAI", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Loading model and explainers...")
def load_all():
    X_train, X_test, _, _ = load_split_data()
    model = joblib.load("models/rf_model.joblib")
    imputer = joblib.load("models/imputer.joblib")
    shap_exp = build_shap_explainer(model)
    values, base = compute_shap(shap_exp, X_test)
    return {"features": list(X_train.columns), "X_test": X_test, "model": model,
            "imputer": imputer, "lime": build_lime_explainer(X_train),
            "shap": shap_exp, "test_values": values, "test_base": base}


def read_inputs(features):
    """Sidebar form. Optional fields have an 'unknown' checkbox. Returns (values, submitted)."""
    vals = {}
    with st.sidebar.form("patient_form"):
        st.header("Patient details")
        for name in features:
            label, lo, hi, default, step, optional = FIELDS[name]
            unknown = optional and st.checkbox(f"{label.split(' (')[0]}: unknown", key=f"u_{name}")
            v = st.number_input(label, min_value=type(step)(lo), max_value=type(step)(hi),
                                value=type(step)(default), step=step, key=name,
                                disabled=bool(unknown))
            vals[name] = np.nan if unknown else float(v)
        submitted = st.form_submit_button("Predict and explain", type="primary")
    return vals, submitted


def main():
    st.title("🩺 ClinXAI: Transparent Diabetes Risk Explanations")
    st.error(DISCLAIMER)
    st.caption("Random Forest trained on the Pima Indians Diabetes dataset (adult women of Pima "
               "heritage), so results do not generalise to other groups. Unknown fields are "
               "filled with the training median.")

    try:
        b = load_all()
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}. Keep app_streamlit.py in the project folder with "
                 "src/, models/ and data/ beside it.")
        st.stop()

    feats = b["features"]
    vals, submitted = read_inputs(feats)
    tab_pred, tab_lime, tab_shap, tab_global = st.tabs(
        ["Prediction", "LIME explanation", "SHAP explanation", "Global importance"])

    row_df = row = was_missing = None
    if submitted:
        raw_df = pd.DataFrame([vals], columns=feats)
        was_missing = raw_df.iloc[0].isna()
        row_df = pd.DataFrame(b["imputer"].transform(raw_df), columns=feats)
        row = row_df.iloc[0]
        prob = float(b["model"].predict_proba(row_df)[0, 1])

    with tab_pred:
        if row is None:
            st.info("Enter patient details in the sidebar and press **Predict and explain**.")
        else:
            c1, c2 = st.columns([1, 2])
            c1.metric("Predicted diabetes risk", f"{prob:.1%}")
            c1.write("Model output at 0.5 threshold: **" +
                     ("Diabetes (positive)" if prob > 0.5 else "No diabetes (negative)") + "**")
            c1.progress(min(max(prob, 0.0), 1.0))
            table = pd.DataFrame({"Feature": feats, "Value used": row.round(2).values,
                                  "Filled with median?": ["Yes" if m else "No" for m in was_missing]})
            c2.dataframe(table, hide_index=True, use_container_width=True)
        if os.path.exists("outputs/metrics.json"):
            m = json.load(open("outputs/metrics.json"))["test"]
            st.caption(f"Model test-set performance (154 held-out patients): accuracy {m['accuracy']}, "
                       f"recall {m['recall']}, ROC-AUC {m['roc_auc']}.")

    with tab_lime:
        if row is None:
            st.info("Run a prediction first.")
        else:
            st.write("LIME fits a simple local model around this patient. "
                     "Positive weights push toward diabetes, negative weights away.")
            with st.spinner("Running LIME..."):
                exp = explain_patient(b["lime"], b["model"], row, num_samples=3000)
            fig = exp.as_pyplot_figure()
            fig.set_size_inches(8, 4.5)
            fig.tight_layout()
            st.pyplot(fig)
            st.dataframe(pd.DataFrame(exp.as_list(), columns=["Rule", "Weight"]).round(4),
                         hide_index=True)
            plt.close(fig)

    with tab_shap:
        if row is None:
            st.info("Run a prediction first.")
        else:
            values, base = compute_shap(b["shap"], row_df)
            st.write(f"SHAP starts from a base value of {base:.2f} and adds each feature's "
                     "contribution to reach this patient's prediction.")
            st.subheader("Waterfall plot")
            plt.close("all")
            shap.plots.waterfall(make_explanation(values[0], base, row), max_display=8, show=False)
            st.pyplot(plt.gcf())
            plt.close("all")
            st.subheader("Force plot")
            try:
                shap.force_plot(base, values[0], row.round(2).values, feature_names=feats,
                                matplotlib=True, show=False, figsize=(14, 3.5))
                st.pyplot(plt.gcf())
            except Exception as e:
                st.warning(f"Force plot not available here: {type(e).__name__}: {e}")
            plt.close("all")

    with tab_global:
        st.write("Global importance = mean absolute SHAP value over the 154 test patients.")
        imp = global_importance(b["test_values"], feats)
        st.bar_chart(imp.rename("Mean |SHAP|"))
        plt.close("all")
        shap.summary_plot(b["test_values"], b["X_test"], show=False)
        st.pyplot(plt.gcf())
        plt.close("all")


main()
