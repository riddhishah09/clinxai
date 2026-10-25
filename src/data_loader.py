"""Stage 1: load the Pima Indians Diabetes dataset (educational use only)."""
import pandas as pd
from sklearn.datasets import fetch_openml

COLUMN_NAMES = {
    "preg": "Pregnancies",
    "plas": "Glucose",
    "pres": "BloodPressure",
    "skin": "SkinThickness",
    "insu": "Insulin",
    "mass": "BMI",
    "pedi": "DiabetesPedigreeFunction",
    "age": "Age",
}

FEATURE_DESCRIPTIONS = {
    "Pregnancies": "Number of pregnancies",
    "Glucose": "Plasma glucose concentration (2h oral glucose tolerance test)",
    "BloodPressure": "Diastolic blood pressure (mm Hg)",
    "SkinThickness": "Triceps skin-fold thickness (mm)",
    "Insulin": "2-hour serum insulin (mu U/ml)",
    "BMI": "Body mass index (kg/m^2)",
    "DiabetesPedigreeFunction": "Diabetes family-history score",
    "Age": "Age in years",
}

TARGET = "Outcome"


def load_diabetes_data():
    """Download the dataset from OpenML and return a clean-named DataFrame."""
    raw = fetch_openml(name="diabetes", version=1, as_frame=True)
    df = raw.frame.copy()
    df[TARGET] = (df["class"] == "tested_positive").astype(int)
    df = df.drop(columns=["class"]).rename(columns=COLUMN_NAMES)
    return df
