"""Stage 2: cleaning helpers (educational use only)."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Columns where a value of 0 is biologically impossible
IMPOSSIBLE_ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def replace_impossible_zeros(df):
    """Return a copy of df with impossible zeros replaced by NaN (missing)."""
    clean = df.copy()
    clean[IMPOSSIBLE_ZERO_COLS] = clean[IMPOSSIBLE_ZERO_COLS].replace(0, np.nan)
    return clean


def remove_duplicates(df):
    """Drop exact duplicate rows."""
    return df.drop_duplicates().reset_index(drop=True)


def build_imputer():
    """Median imputer that returns a DataFrame. Fit it on TRAINING data only (Stage 3)."""
    imputer = SimpleImputer(strategy="median")
    imputer.set_output(transform="pandas")
    return imputer
