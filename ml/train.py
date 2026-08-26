from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "ml" / "models"

MODEL_A_PATH = MODEL_DIR / "behavior_model.pkl"
MODEL_B_PATH = MODEL_DIR / "behavior_network_model.pkl"


# ---------------------------------------------------------
# FEATURE SETS
# ---------------------------------------------------------

BEHAVIOR_FEATURES = [
    "total_orders",
    "returned_orders",
    "total_spend",
    "avg_order_value",
    "max_order_value",
    "total_return_cost",
    "avg_return_cost",
    "avg_days_to_return",
    "min_days_to_return",
    "account_age_days",
    "unique_products",
    "unique_categories",
    "unique_payment_methods",
    "return_rate",
    "return_cost_ratio",
    "rapid_returns",
    "rapid_return_ratio",
    "activity_span_days",
    "orders_per_30_days",
]


NETWORK_FEATURES = [
    "accounts_same_device",
    "accounts_same_address",
    "accounts_same_payment",
]


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data():
    df = pd.read_csv(DATA_PATH)

    print(
        f"Loaded feature dataset: "
        f"{len(df)} customers"
    )

    return df


# ---------------------------------------------------------
# SPLIT DATA
# ---------------------------------------------------------

def split_data(df):

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["is_abuse"],
        random_state=42,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["is_abuse"],
        random_state=42,
    )

    print("\n--- DATA SPLIT ---")

    print(
        f"Train: {len(train_df)}"
    )

    print(
        f"Validation: {len(val_df)}"
    )

    print(
        f"Test: {len(test_df)}"
    )

    return train_df, val_df, test_df


# ---------------------------------------------------------
# CLASS WEIGHT
# ---------------------------------------------------------

def calculate_scale_pos_weight(y):

    negatives = (y == 0).sum()
    positives = (y == 1).sum()

    return negatives / positives


# ---------------------------------------------------------
# BUILD MODEL
# ---------------------------------------------------------

def build_model(scale_pos_weight):

    return XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )


# ---------------------------------------------------------
# THRESHOLD SELECTION
# ---------------------------------------------------------

def select_threshold(
    y_true,
    probabilities,
):

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities,
        )
    )

    f1_scores = (
        2
        * precision[:-1]
        * recall[:-1]
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-10
        )
    )

    best_index = np.argmax(
        f1_scores
    )

    best_threshold = (
        thresholds[
            best_index
        ]
    )

    return float(
        best_threshold
    )


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def evaluate_model(
    model,
    X,
    y,
    threshold,
    name,
):

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    print(
        f"\n--- {name} ---"
    )

    print(
        f"Threshold: {threshold:.3f}"
    )

    print(
        f"Precision: {precision:.3f}"
    )

    print(
        f"Recall: {recall:.3f}"
    )

    print(
        f"F1: {f1:.3f}"
    )

    print(
        f"PR-AUC: {pr_auc:.3f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            y,
            predictions,
        )
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y,
            predictions,
            digits=3,
            zero_division=0,
        )
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
    }


# ---------------------------------------------------------
# TRAIN ONE MODEL
# ---------------------------------------------------------

def train_one_model(
    train_df,
    val_df,
    test_df,
    features,
    model_name,
    output_path,
):

    X_train = train_df[
        features
    ]

    y_train = train_df[
        "is_abuse"
    ]

    X_val = val_df[
        features
    ]

    y_val = val_df[
        "is_abuse"
    ]

    X_test = test_df[
        features
    ]

    y_test = test_df[
        "is_abuse"
    ]

    scale_pos_weight = (
        calculate_scale_pos_weight(
            y_train
        )
    )

    print(
        f"\nTraining {model_name}"
    )

    print(
        "scale_pos_weight:",
        round(
            scale_pos_weight,
            2,
        ),
    )

    model = build_model(
        scale_pos_weight
    )

    model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------
    # Select threshold ONLY on validation
    # ---------------------------------

    val_probabilities = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    threshold = (
        select_threshold(
            y_val,
            val_probabilities,
        )
    )

    print(
        f"Chosen validation threshold: "
        f"{threshold:.3f}"
    )

    # ---------------------------------
    # Evaluate validation
    # ---------------------------------

    evaluate_model(
        model,
        X_val,
        y_val,
        threshold,
        f"{model_name} VALIDATION",
    )

    # ---------------------------------
    # Evaluate untouched test set
    # ---------------------------------

    test_metrics = evaluate_model(
        model,
        X_test,
        y_test,
        threshold,
        f"{model_name} TEST",
    )

    joblib.dump(
        {
            "model": model,
            "features": features,
            "threshold": threshold,
        },
        output_path,
    )

    print(
        f"\nSaved model to:"
    )

    print(
        output_path
    )

    return test_metrics


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    train_df, val_df, test_df = (
        split_data(df)
    )

    # -----------------------------------------------------
    # MODEL A
    # Behaviour only
    # -----------------------------------------------------

    behavior_metrics = (
        train_one_model(
            train_df,
            val_df,
            test_df,
            BEHAVIOR_FEATURES,
            "MODEL A - BEHAVIOR",
            MODEL_A_PATH,
        )
    )

    # -----------------------------------------------------
    # MODEL B
    # Behaviour + identity-link features
    # -----------------------------------------------------

    all_features = (
        BEHAVIOR_FEATURES
        + NETWORK_FEATURES
    )

    network_metrics = (
        train_one_model(
            train_df,
            val_df,
            test_df,
            all_features,
            "MODEL B - BEHAVIOR + NETWORK",
            MODEL_B_PATH,
        )
    )

    print(
        "\n=============================="
    )

    print(
        "FINAL TEST COMPARISON"
    )

    print(
        "=============================="
    )

    print(
        "\nBehavior model:"
    )

    print(
        behavior_metrics
    )

    print(
        "\nBehavior + network model:"
    )

    print(
        network_metrics
    )


if __name__ == "__main__":
    main()