"""ClinXAI Streamlit web interface (educational use only).
Run from the project folder:  streamlit run app_streamlit.py
Needs: src/, models/rf_model.joblib, models/imputer.joblib, data/X_train.csv, data/X_test.csv
Optional: .streamlit/config.toml (theme colours)
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

DISCLAIMER = ("Educational project only. This is not a medical device and must not be used "
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

# ---- Colour tokens (also used by the matplotlib figures) ----
INK, TEAL, MINT = "#0B2E36", "#0E6F73", "#DDF1EE"
LOW, MID, HIGH = "#1F9D8B", "#E0A100", "#D6403F"

st.set_page_config(page_title="ClinXAI | Transparent Diabetes Risk", page_icon="🩺",
                   layout="wide", initial_sidebar_state="expanded")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Figtree:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'Figtree', sans-serif; color: {INK}; }}
.stApp {{ background: #F2F6F7; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1.6rem; max-width: 1200px; }}

h1, h2, h3, .cx-display {{ font-family: 'Fraunces', serif !important; color: {INK}; letter-spacing: -0.01em; }}
h3 {{ font-size: 1.35rem !important; }}

/* Hero */
.cx-hero {{ background: {INK}; border-radius: 20px; padding: 2.2rem 2.4rem 1.6rem; color: #fff;
            position: relative; overflow: hidden; margin-bottom: 1rem; }}
.cx-hero h1 {{ color: #fff !important; font-size: 2.6rem; margin: 0 0 .4rem 0; line-height: 1.1; }}
.cx-hero p {{ color: #B9D3D6; font-size: 1.05rem; max-width: 620px; margin: 0; line-height: 1.55; }}
.cx-hero svg {{ position: absolute; right: 0; bottom: 0; width: 55%; height: 100%; opacity: .9; }}
.cx-pills {{ margin-top: 1.1rem; position: relative; z-index: 2; }}
.cx-pill {{ display: inline-block; background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.2);
            color: #fff; border-radius: 999px; padding: .25rem .8rem; font-size: .85rem; margin-right: .4rem; }}
.cx-hero > div {{ position: relative; z-index: 2; }}

/* Notice */
.cx-notice {{ background: #FFF6DD; border-left: 5px solid {MID}; border-radius: 10px;
              padding: .75rem 1rem; font-size: .93rem; margin-bottom: 1.1rem; }}

/* Cards */
.cx-card {{ background: #fff; border: 1px solid #D9E4E7; border-radius: 16px; padding: 1.2rem 1.4rem;
            height: 100%; }}
.cx-card h4 {{ margin: 0 0 .35rem 0; font-size: 1.05rem; color: {INK}; }}
.cx-card p {{ margin: 0; color: #476067; font-size: .95rem; line-height: 1.5; }}
.cx-step {{ width: 30px; height: 30px; border-radius: 50%; background: {MINT}; color: {TEAL};
            font-weight: 700; display: flex; align-items: center; justify-content: center; margin-bottom: .6rem; }}

/* Risk result */
.cx-risk {{ background: #fff; border: 1px solid #D9E4E7; border-radius: 16px; padding: 1.3rem 1.4rem; }}
.cx-risk .label {{ color: #476067; font-size: .9rem; margin-bottom: .2rem; }}
.cx-risk .value {{ font-family: 'Fraunces', serif; font-size: 3.4rem; font-weight: 700; line-height: 1; }}
.cx-badge {{ display: inline-block; border-radius: 999px; padding: .25rem .8rem; font-weight: 600;
             font-size: .85rem; color: #fff; margin-top: .6rem; }}
.cx-track {{ position: relative; height: 14px; border-radius: 999px; margin: 1.3rem 0 .3rem;
             background: linear-gradient(90deg, {LOW} 0%, {LOW} 35%, {MID} 35%, {MID} 65%, {HIGH} 65%, {HIGH} 100%); }}
.cx-mark {{ position: absolute; top: -6px; width: 6px; height: 26px; background: {INK};
            border: 2px solid #fff; border-radius: 4px; transform: translateX(-50%); }}
.cx-tick {{ position: absolute; top: -4px; width: 2px; height: 22px; background: rgba(255,255,255,.9); }}
.cx-scale {{ display: flex; justify-content: space-between; font-size: .75rem; color: #6A8288; }}

/* KPI */
.cx-kpi {{ background: #fff; border: 1px solid #D9E4E7; border-radius: 14px; padding: .9rem 1.1rem; }}
.cx-kpi .n {{ font-family: 'Fraunces', serif; font-size: 1.8rem; font-weight: 700; color: {TEAL}; }}
.cx-kpi .t {{ font-size: .85rem; color: #476067; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: .3rem; border-bottom: 1px solid #D9E4E7; }}
.stTabs [data-baseweb="tab"] {{ height: 46px; padding: 0 1.1rem; border-radius: 10px 10px 0 0;
                               font-weight: 600; color: #476067; }}
.stTabs [aria-selected="true"] {{ color: {TEAL} !important; background: #fff; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {TEAL}; height: 3px; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background: #fff; border-right: 1px solid #D9E4E7; }}
section[data-testid="stSidebar"] h2 {{ font-size: 1.4rem; }}
div[data-testid="stForm"] {{ border: none; padding: 0; }}
.stButton > button, div[data-testid="stFormSubmitButton"] > button {{
    background: {TEAL}; color: #fff; border: none; border-radius: 12px; font-weight: 700;
    padding: .65rem 1rem; width: 100%; transition: background .15s; }}
.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {{ background: {INK}; color: #fff; }}
.stButton > button:focus-visible, div[data-testid="stFormSubmitButton"] > button:focus-visible {{
    outline: 3px solid {MID}; outline-offset: 2px; }}

/* Dataframes and images */
div[data-testid="stDataFrame"] {{ border: 1px solid #D9E4E7; border-radius: 12px; overflow: hidden; }}
div[data-testid="stImage"] img, div[data-testid="stPyplotRootElement"] img {{ border-radius: 12px; }}

@media (max-width: 700px) {{
  .cx-hero {{ padding: 1.4rem; }} .cx-hero h1 {{ font-size: 1.8rem; }} .cx-hero svg {{ display: none; }}
  .cx-risk .value {{ font-size: 2.6rem; }}
}}
</style>
"""

ECG_SVG = (
    '<svg viewBox="0 0 600 200" preserveAspectRatio="xMaxYMax slice" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M0 120 H140 L165 120 L185 50 L215 170 L240 85 L258 120 H360 L385 120 L405 70 '
    'L430 150 L450 120 H600" fill="none" stroke="#3FC8B6" stroke-width="3" stroke-linecap="round" '
    'stroke-linejoin="round" opacity=".55"/></svg>'
)


def show_df(df, **kw):
    """Full-width dataframe that works on old and new Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def style_fig(fig):
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
    return fig


def hero():
    st.markdown(
        f'<div class="cx-hero">{ECG_SVG}<div>'
        '<h1>ClinXAI</h1>'
        '<p>See not just <b>what</b> a Random Forest predicts about diabetes risk, '
        'but <b>why</b>, explained side by side with LIME and SHAP.</p>'
        '<div class="cx-pills"><span class="cx-pill">Random Forest</span>'
        '<span class="cx-pill">LIME</span><span class="cx-pill">SHAP</span>'
        '<span class="cx-pill">Pima Indians dataset</span></div>'
        '</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="cx-notice"><b>Please note:</b> {DISCLAIMER} The model was trained on '
                'adult women of Pima heritage, so results do not generalise to other groups. '
                'Unknown fields are filled with the training median.</div>', unsafe_allow_html=True)


def risk_card(prob):
    if prob < 0.35:
        colour, band = LOW, "Lower risk band"
    elif prob <= 0.65:
        colour, band = MID, "Borderline band"
    else:
        colour, band = HIGH, "Higher risk band"
    outcome = "Diabetes (positive)" if prob > 0.5 else "No diabetes (negative)"
    pct = min(max(prob, 0.0), 1.0) * 100
    st.markdown(
        f'<div class="cx-risk"><div class="label">Predicted diabetes risk</div>'
        f'<div class="value" style="color:{colour}">{prob:.1%}</div>'
        f'<span class="cx-badge" style="background:{colour}">{band}</span>'
        f'<div class="cx-track"><div class="cx-tick" style="left:50%"></div>'
        f'<div class="cx-mark" style="left:{pct:.1f}%"></div></div>'
        f'<div class="cx-scale"><span>0%</span><span>0.5 threshold</span><span>100%</span></div>'
        f'<p style="margin:.9rem 0 0 0;font-size:.95rem">Model output: <b>{outcome}</b></p>'
        f'<p style="margin:.3rem 0 0 0;font-size:.78rem;color:#6A8288">Colour bands are for display only.</p>'
        f'</div>', unsafe_allow_html=True)


def how_it_works():
    cols = st.columns(3, gap="medium")
    steps = [("1", "Enter patient details",
              "Use the sidebar. Tick <b>unknown</b> for any optional value and the training median is used."),
             ("2", "Get a risk estimate",
              "The Random Forest returns a probability of diabetes for this patient."),
             ("3", "Read the explanations",
              "LIME and SHAP show which features pushed the prediction up or down.")]
    for col, (n, t, d) in zip(cols, steps):
        col.markdown(f'<div class="cx-card"><div class="cx-step">{n}</div><h4>{t}</h4><p>{d}</p></div>',
                     unsafe_allow_html=True)


def kpi_row():
    if not os.path.exists("outputs/metrics.json"):
        return
    m = json.load(open("outputs/metrics.json"))["test"]
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("### Model performance")
    st.caption("Measured on the 154 held-out test patients.")
    cols = st.columns(3, gap="medium")
    for col, (k, t) in zip(cols, [("accuracy", "Accuracy"), ("recall", "Recall"), ("roc_auc", "ROC-AUC")]):
        col.markdown(f'<div class="cx-kpi"><div class="n">{m[k]}</div><div class="t">{t}</div></div>',
                     unsafe_allow_html=True)


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
        st.caption("Enter values, then press the button at the bottom.")
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
    st.markdown(CSS, unsafe_allow_html=True)
    hero()

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
            how_it_works()
        else:
            c1, c2 = st.columns([1, 1.5], gap="large")
            with c1:
                risk_card(prob)
            with c2:
                st.markdown("### Values used by the model")
                table = pd.DataFrame({"Feature": feats, "Value used": row.round(2).values,
                                      "Filled with median?": ["Yes" if m else "No" for m in was_missing]})
                show_df(table, hide_index=True)
        kpi_row()

    with tab_lime:
        if row is None:
            st.info("Press **Predict and explain** in the sidebar to see the LIME explanation.")
        else:
            st.markdown("### LIME: a simple model around this patient")
            st.write("Positive weights push toward diabetes, negative weights push away.")
            with st.spinner("Running LIME..."):
                exp = explain_patient(b["lime"], b["model"], row, num_samples=3000)
            fig = style_fig(exp.as_pyplot_figure())
            fig.set_size_inches(8, 4.5)
            fig.tight_layout()
            st.pyplot(fig)
            show_df(pd.DataFrame(exp.as_list(), columns=["Rule", "Weight"]).round(4), hide_index=True)
            plt.close(fig)

    with tab_shap:
        if row is None:
            st.info("Press **Predict and explain** in the sidebar to see the SHAP explanation.")
        else:
            values, base = compute_shap(b["shap"], row_df)
            st.markdown("### SHAP: each feature's contribution")
            st.write(f"SHAP starts from a base value of **{base:.2f}** and adds each feature's "
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
        st.markdown("### Which features matter most overall?")
        st.write("Global importance is the mean absolute SHAP value over the 154 test patients.")
        imp = global_importance(b["test_values"], feats)
        g1, g2 = st.columns(2, gap="large")
        with g1:
            st.markdown("**Mean |SHAP| per feature**")
            st.bar_chart(imp.rename("Mean |SHAP|"), color=TEAL)
        with g2:
            st.markdown("**SHAP beeswarm**")
            plt.close("all")
            shap.summary_plot(b["test_values"], b["X_test"], show=False)
            st.pyplot(plt.gcf())
            plt.close("all")

    st.markdown(f'<p style="text-align:center;color:#6A8288;font-size:.85rem;margin-top:2.5rem">'
                f'ClinXAI &middot; college micro-project on interpretable machine learning &middot; '
                f'educational use only</p>', unsafe_allow_html=True)


main()
