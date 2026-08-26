from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "returnshield_labeled.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_features.csv"
)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=[
            "order_date",
            "account_created_date",
        ],
    )

    print(
        f"Loaded labeled dataset: "
        f"{len(df)} orders"
    )

    return df


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:

    denominator = denominator.replace(0, np.nan)

    result = numerator / denominator

    return result.fillna(0)


def build_customer_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    # -----------------------------------------------------
    # BASIC CUSTOMER BEHAVIOUR
    # -----------------------------------------------------

    customer = (
        df.groupby("customer_id")
        .agg(
            total_orders=(
                "order_id",
                "nunique",
            ),
            returned_orders=(
                "is_returned",
                "sum",
            ),
            total_spend=(
                "order_value",
                "sum",
            ),
            avg_order_value=(
                "order_value",
                "mean",
            ),
            max_order_value=(
                "order_value",
                "max",
            ),
            total_return_cost=(
                "return_cost",
                "sum",
            ),
            avg_return_cost=(
                "return_cost",
                "mean",
            ),
            avg_days_to_return=(
                "days_to_return",
                "mean",
            ),
            min_days_to_return=(
                "days_to_return",
                "min",
            ),
            account_age_days=(
                "account_age_days",
                "max",
            ),
            first_order_date=(
                "order_date",
                "min",
            ),
            last_order_date=(
                "order_date",
                "max",
            ),
            unique_products=(
                "product_id",
                "nunique",
            ),
            unique_categories=(
                "product_category",
                "nunique",
            ),
            unique_payment_methods=(
                "payment_method",
                "nunique",
            ),
        )
        .reset_index()
    )

    # -----------------------------------------------------
    # RATIOS
    # -----------------------------------------------------

    customer["return_rate"] = safe_ratio(
        customer["returned_orders"],
        customer["total_orders"],
    )

    customer["return_cost_ratio"] = safe_ratio(
        customer["total_return_cost"],
        customer["total_spend"],
    )

    # -----------------------------------------------------
    # RAPID RETURN FEATURES
    # -----------------------------------------------------

    returned_df = df[
        df["is_returned"] == 1
    ].copy()

    rapid_return_counts = (
        returned_df[
            returned_df["days_to_return"] <= 5
        ]
        .groupby("customer_id")
        .size()
        .rename("rapid_returns")
    )

    customer = customer.merge(
        rapid_return_counts,
        on="customer_id",
        how="left",
    )

    customer["rapid_returns"] = (
        customer["rapid_returns"]
        .fillna(0)
        .astype(int)
    )

    customer["rapid_return_ratio"] = safe_ratio(
        customer["rapid_returns"],
        customer["returned_orders"],
    )

    # -----------------------------------------------------
    # CUSTOMER ACTIVITY WINDOW
    # -----------------------------------------------------

    customer["activity_span_days"] = (
        customer["last_order_date"]
        - customer["first_order_date"]
    ).dt.days

    customer[
        "orders_per_30_days"
    ] = safe_ratio(
        customer["total_orders"] * 30,
        customer["activity_span_days"] + 1,
    )

    # -----------------------------------------------------
    # GRAPH-LIKE IDENTITY FEATURES
    #
    # These are not full graph features yet.
    # They simply count how many customers share
    # each identity.
    # -----------------------------------------------------

    customer_device = (
        df[
            [
                "customer_id",
                "device_id",
            ]
        ]
        .drop_duplicates()
    )

    device_counts = (
        customer_device
        .groupby("device_id")[
            "customer_id"
        ]
        .nunique()
        .rename(
            "accounts_same_device"
        )
    )

    customer_device = customer_device.merge(
        device_counts,
        on="device_id",
        how="left",
    )

    device_feature = (
        customer_device
        .groupby("customer_id")[
            "accounts_same_device"
        ]
        .max()
        .reset_index()
    )

    customer = customer.merge(
        device_feature,
        on="customer_id",
        how="left",
    )

    # -----------------------------------------------------

    customer_address = (
        df[
            [
                "customer_id",
                "shipping_address_id",
            ]
        ]
        .drop_duplicates()
    )

    address_counts = (
        customer_address
        .groupby(
            "shipping_address_id"
        )["customer_id"]
        .nunique()
        .rename(
            "accounts_same_address"
        )
    )

    customer_address = (
        customer_address.merge(
            address_counts,
            on="shipping_address_id",
            how="left",
        )
    )

    address_feature = (
        customer_address
        .groupby("customer_id")[
            "accounts_same_address"
        ]
        .max()
        .reset_index()
    )

    customer = customer.merge(
        address_feature,
        on="customer_id",
        how="left",
    )

    # -----------------------------------------------------

    customer_payment = (
        df[
            [
                "customer_id",
                "payment_instrument_id",
            ]
        ]
        .drop_duplicates()
    )

    payment_counts = (
        customer_payment
        .groupby(
            "payment_instrument_id"
        )["customer_id"]
        .nunique()
        .rename(
            "accounts_same_payment"
        )
    )

    customer_payment = (
        customer_payment.merge(
            payment_counts,
            on="payment_instrument_id",
            how="left",
        )
    )

    payment_feature = (
        customer_payment
        .groupby("customer_id")[
            "accounts_same_payment"
        ]
        .max()
        .reset_index()
    )

    customer = customer.merge(
        payment_feature,
        on="customer_id",
        how="left",
    )

    # -----------------------------------------------------
    # LABEL
    # -----------------------------------------------------

    labels = (
        df.groupby("customer_id")
        .agg(
            is_abuse=(
                "is_abuse",
                "max",
            ),
            abuse_type=(
                "abuse_type",
                "first",
            ),
        )
        .reset_index()
    )

    customer = customer.merge(
        labels,
        on="customer_id",
        how="left",
    )

    # Dates are not directly fed into the model.
    customer = customer.drop(
        columns=[
            "first_order_date",
            "last_order_date",
        ]
    )

    return customer


def validate_features(
    df: pd.DataFrame,
) -> None:

    print(
        "\n--- FEATURE TABLE ---"
    )

    print(
        "Customers:",
        len(df),
    )

    print(
        "Abusive customers:",
        df["is_abuse"].sum(),
    )

    print(
        "Legitimate customers:",
        (
            df["is_abuse"] == 0
        ).sum(),
    )

    print(
        "\nColumns:"
    )

    for column in df.columns:
        print(
            " -",
            column,
        )

    print(
        "\nMissing values:"
    )

    missing = (
        df.isnull()
        .sum()
    )

    print(
        missing[
            missing > 0
        ]
    )

    print(
        "\nMean return rate:"
    )

    print(
        df.groupby(
            "is_abuse"
        )["return_rate"]
        .mean()
    )

    print(
        "\nMean shared-device count:"
    )

    print(
        df.groupby(
            "is_abuse"
        )[
            "accounts_same_device"
        ]
        .mean()
    )


def main():

    df = load_data()

    features = build_customer_features(
        df
    )

    validate_features(
        features
    )

    features.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nSaved customer features to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()