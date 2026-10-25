"""Stage 3: train/test split and leak-free imputation (educational use only)."""
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import build_imputer

TARGET = "Outcome"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def split_data(df):
    """Stratified split. Returns X_train, X_test, y_train, y_test (still with NaN)."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def impute_train_test(X_train, X_test):
    """Fit median imputer on TRAIN only, transform both. Returns (X_train, X_test, imputer)."""
    imputer = build_imputer()
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    return X_train_imp, X_test_imp, imputer
