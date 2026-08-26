from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_features.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "behavior_network_model.pkl"
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data():

    df = pd.read_csv(DATA_PATH)

    print(
        f"Loaded dataset: {len(df)} customers"
    )

    return df


# ---------------------------------------------------------
# RECREATE SAME TEST SPLIT
# ---------------------------------------------------------

def get_test_set(df):

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

    return test_df.copy()


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

def load_model():

    package = joblib.load(MODEL_PATH)

    model = package["model"]
    features = package["features"]
    threshold = package["threshold"]

    return model, features, threshold


# ---------------------------------------------------------
# CREATE PREDICTIONS
# ---------------------------------------------------------

def predict(
    test_df,
    model,
    features,
    threshold,
):

    probabilities = (
        model.predict_proba(
            test_df[features]
        )[:, 1]
    )

    test_df[
        "risk_probability"
    ] = probabilities

    test_df[
        "predicted_abuse"
    ] = (
        probabilities
        >= threshold
    ).astype(int)

    return test_df


# ---------------------------------------------------------
# MONETARY METRICS
# ---------------------------------------------------------

def calculate_money_metrics(df):

    # We use return-related cost as the value
    # exposed to return abuse in this prototype.

    true_positive = df[
        (df["is_abuse"] == 1)
        & (df["predicted_abuse"] == 1)
    ]

    false_negative = df[
        (df["is_abuse"] == 1)
        & (df["predicted_abuse"] == 0)
    ]

    false_positive = df[
        (df["is_abuse"] == 0)
        & (df["predicted_abuse"] == 1)
    ]

    true_negative = df[
        (df["is_abuse"] == 0)
        & (df["predicted_abuse"] == 0)
    ]

    # -----------------------------------------------------
    # VALUE CALCULATIONS
    # -----------------------------------------------------

    total_abusive_value = (
        df.loc[
            df["is_abuse"] == 1,
            "total_return_cost",
        ]
        .sum()
    )

    detected_abusive_value = (
        true_positive[
            "total_return_cost"
        ]
        .sum()
    )

    missed_abusive_value = (
        false_negative[
            "total_return_cost"
        ]
        .sum()
    )

    legitimate_flagged_value = (
        false_positive[
            "total_return_cost"
        ]
        .sum()
    )

    # -----------------------------------------------------
    # RECOVERY RATE
    # -----------------------------------------------------

    if total_abusive_value > 0:

        value_detection_rate = (
            detected_abusive_value
            / total_abusive_value
        )

    else:
        value_detection_rate = 0

    return {
        "tp_count": len(true_positive),
        "fp_count": len(false_positive),
        "fn_count": len(false_negative),
        "tn_count": len(true_negative),

        "total_abusive_value":
            total_abusive_value,

        "detected_abusive_value":
            detected_abusive_value,

        "missed_abusive_value":
            missed_abusive_value,

        "legitimate_flagged_value":
            legitimate_flagged_value,

        "value_detection_rate":
            value_detection_rate,
    }


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------
def print_money_metrics(metrics):

    print("\n==============================")
    print("RETURNSHIELD MONETARY IMPACT")
    print("==============================")

    print("\nCASE OUTCOMES")

    print(f"True positives:  {metrics['tp_count']}")
    print(f"False positives: {metrics['fp_count']}")
    print(f"False negatives: {metrics['fn_count']}")
    print(f"True negatives:  {metrics['tn_count']}")

    print("\nFINANCIAL IMPACT")

    total_abusive = metrics["total_abusive_value"]
    detected = metrics["detected_abusive_value"]
    missed = metrics["missed_abusive_value"]
    legitimate_flagged = metrics["legitimate_flagged_value"]
    detection_rate = metrics["value_detection_rate"]

    print(
        f"Total abusive return value: ₹{total_abusive:,.2f}"
    )

    print(
        f"Abusive value detected: ₹{detected:,.2f}"
    )

    print(
        f"Abusive value missed: ₹{missed:,.2f}"
    )

    print(
        "Legitimate value incorrectly flagged: "
        f"₹{legitimate_flagged:,.2f}"
    )

    print(
        f"Value detection rate: {detection_rate:.2%}"
    )

# ---------------------------------------------------------
# EXPORT CASES
# ---------------------------------------------------------

def export_predictions(df):

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "test_predictions.csv"
    )

    columns = [
        "customer_id",
        "is_abuse",
        "abuse_type",
        "risk_probability",
        "predicted_abuse",
        "total_return_cost",
        "return_rate",
        "accounts_same_device",
        "accounts_same_address",
        "accounts_same_payment",
    ]

    df[
        columns
    ].sort_values(
        "risk_probability",
        ascending=False,
    ).to_csv(
        output_path,
        index=False,
    )

    print(
        "\nSaved test predictions to:"
    )

    print(
        output_path
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    df = load_data()

    test_df = get_test_set(df)

    model, features, threshold = (
        load_model()
    )

    test_df = predict(
        test_df,
        model,
        features,
        threshold,
    )

    metrics = (
        calculate_money_metrics(
            test_df
        )
    )

    print_money_metrics(
        metrics
    )

    export_predictions(
        test_df
    )


if __name__ == "__main__":
    main()