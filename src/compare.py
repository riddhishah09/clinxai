"""Stage 7: compare LIME and SHAP (educational use only)."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def feature_from_rule(rule, feature_names):
    """LIME rules look like '99.00 < Glucose <= 117.00'. Find which feature the rule is about."""
    matches = [f for f in feature_names if f in rule]
    return max(matches, key=len) if matches else None


def lime_pairs_to_series(pairs, feature_names):
    """Turn LIME's (rule, weight) pairs into one weight per feature."""
    out = {f: 0.0 for f in feature_names}
    for rule, weight in pairs:
        f = feature_from_rule(rule, feature_names)
        if f is not None:
            out[f] += float(weight)
    return pd.Series(out)


def compare_vectors(lime, shap, top_k=3):
    """Compare two importance vectors (pandas Series with the same feature index)."""
    shap = shap.reindex(lime.index)
    rho = spearmanr(lime.abs(), shap.abs())[0]
    top_l = list(lime.abs().nlargest(top_k).index)
    top_s = list(shap.abs().nlargest(top_k).index)
    return {
        "spearman_abs": round(float(rho), 3),
        "top3_overlap": len(set(top_l) & set(top_s)),
        "top1_lime": top_l[0],
        "top1_shap": top_s[0],
        "top1_same": bool(top_l[0] == top_s[0]),
        "sign_agree": int((np.sign(lime) == np.sign(shap)).sum()),
        "n_features": int(len(lime)),
    }


def build_case_table(lime_rows, shap_rows):
    """Merge one case's LIME rows and SHAP rows into one table, with ranks and flags."""
    feats = list(shap_rows["feature"])
    lime_w = lime_pairs_to_series(list(zip(lime_rows["rule"], lime_rows["weight"])), feats)
    rule_text = {}
    for rule in lime_rows["rule"]:
        f = feature_from_rule(rule, feats)
        if f is not None:
            rule_text[f] = rule
    t = shap_rows.set_index("feature")[["value", "shap"]].copy()
    t["lime_rule"] = pd.Series(rule_text)
    t["lime_weight"] = lime_w
    t["lime_rank"] = t["lime_weight"].abs().rank(ascending=False, method="min").astype(int)
    t["shap_rank"] = t["shap"].abs().rank(ascending=False, method="min").astype(int)
    t["rank_gap"] = (t["lime_rank"] - t["shap_rank"]).abs()
    t["sign_same"] = np.sign(t["lime_weight"]) == np.sign(t["shap"])
    return t.sort_values("shap_rank")
