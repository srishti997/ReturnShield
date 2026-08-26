from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "returnshield_base.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "returnshield_labeled.csv"
)


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

INDIVIDUAL_ABUSER_RATE = 0.05
COORDINATED_ABUSER_RATE = 0.04
HARD_NEGATIVE_RATE = 0.06


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=[
            "order_date",
            "account_created_date",
        ],
    )

    print(
        f"Loaded base dataset: "
        f"{len(df)} rows"
    )

    return df


# ---------------------------------------------------------
# CUSTOMER SUMMARY
# ---------------------------------------------------------

def build_customer_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        df.groupby("customer_id")
        .agg(
            total_orders=("order_id", "count"),
            returned_orders=("is_returned", "sum"),
            total_spend=("order_value", "sum"),
            total_return_cost=("return_cost", "sum"),
            avg_days_to_return=("days_to_return", "mean"),
            first_order=("order_date", "min"),
            last_order=("order_date", "max"),
        )
        .reset_index()
    )

    summary["return_rate"] = (
        summary["returned_orders"]
        / summary["total_orders"]
    )

    return summary


# ---------------------------------------------------------
# INITIAL LABELS
# ---------------------------------------------------------

def initialize_labels(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df["is_abuse"] = 0
    df["abuse_type"] = "LEGITIMATE"

    return df


# ---------------------------------------------------------
# HARD NEGATIVES
# ---------------------------------------------------------

def inject_hard_negatives(
    df: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:

    eligible = summary[
        summary["total_orders"] >= 3
    ].copy()

    n_hard = max(
        1,
        int(
            len(summary)
            * HARD_NEGATIVE_RATE
        ),
    )

    n_hard = min(
        n_hard,
        len(eligible),
    )

    hard_customers = rng.choice(
        eligible["customer_id"],
        size=n_hard,
        replace=False,
    )

    # Keep them legitimate
    # but make their behavior look suspicious.
    for customer_id in hard_customers:

        customer_rows = df[
            df["customer_id"] == customer_id
        ].index

        # Increase chance that orders
        # look like returns.
        for idx in customer_rows:

            if rng.random() < 0.55:
                df.loc[idx, "is_returned"] = 1
                df.loc[idx, "return_status"] = "Returned"

                if pd.isna(
                    df.loc[idx, "return_reason"]
                ):
                    df.loc[
                        idx,
                        "return_reason"
                    ] = rng.choice(
                        [
                            "Defective",
                            "Wrong Item",
                            "Size Issue",
                        ]
                    )

                if pd.isna(
                    df.loc[idx, "days_to_return"]
                ):
                    df.loc[
                        idx,
                        "days_to_return"
                    ] = int(
                        rng.integers(
                            3,
                            15,
                        )
                    )

        df.loc[
            customer_rows,
            "abuse_type"
        ] = "HIGH_RETURN_LEGITIMATE"

    return df


# ---------------------------------------------------------
# INDIVIDUAL ABUSE
# ---------------------------------------------------------

def inject_individual_abuse(
    df: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, set]:

    eligible = summary[
        summary["total_orders"] >= 2
    ].copy()

    n_abusers = max(
        1,
        int(
            len(summary)
            * INDIVIDUAL_ABUSER_RATE
        ),
    )

    n_abusers = min(
        n_abusers,
        len(eligible),
    )

    abusers = set(
        rng.choice(
            eligible["customer_id"],
            size=n_abusers,
            replace=False,
        )
    )

    for customer_id in abusers:

        customer_rows = df[
            df["customer_id"] == customer_id
        ].index

        for idx in customer_rows:

            # Make returns more frequent
            if rng.random() < 0.7:
                df.loc[idx, "is_returned"] = 1
                df.loc[idx, "return_status"] = "Returned"

                # More aggressive return timing
                df.loc[
                    idx,
                    "days_to_return"
                ] = int(
                    rng.integers(
                        1,
                        6,
                    )
                )

                df.loc[
                    idx,
                    "return_reason"
                ] = rng.choice(
                    [
                        "Changed Mind",
                        "Wrong Item",
                        "Size Issue",
                        "Defective",
                    ]
                )

                # Increase refund-related cost
                base_cost = df.loc[
                    idx,
                    "return_cost"
                ]

                if pd.isna(base_cost):
                    base_cost = 0

                df.loc[
                    idx,
                    "return_cost"
                ] = max(
                    float(base_cost),
                    float(
                        df.loc[
                            idx,
                            "order_value"
                        ]
                        * rng.uniform(
                            0.45,
                            0.85,
                        )
                    ),
                )

        df.loc[
            customer_rows,
            "is_abuse"
        ] = 1

        df.loc[
            customer_rows,
            "abuse_type"
        ] = "INDIVIDUAL_RETURN_ABUSE"

    return df, abusers


# ---------------------------------------------------------
# COORDINATED ABUSE RINGS
# ---------------------------------------------------------

def inject_abuse_rings(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    excluded_customers: set,
) -> tuple[pd.DataFrame, set]:

    eligible = summary[
        ~summary["customer_id"].isin(
            excluded_customers
        )
    ].copy()

    n_ring_customers = max(
        6,
        int(
            len(summary)
            * COORDINATED_ABUSER_RATE
        ),
    )

    n_ring_customers = min(
        n_ring_customers,
        len(eligible),
    )

    selected = list(
        rng.choice(
            eligible["customer_id"],
            size=n_ring_customers,
            replace=False,
        )
    )

    ring_customers = set(selected)

    cursor = 0
    ring_id = 1

    while cursor < len(selected):

        ring_size = int(
            rng.integers(
                3,
                6,
            )
        )

        members = selected[
            cursor:
            cursor + ring_size
        ]

        if len(members) < 2:
            break

        shared_device = (
            f"ABUSE_DEV_{ring_id:03d}"
        )

        shared_address = (
            f"ABUSE_ADDR_{ring_id:03d}"
        )

        shared_payment = (
            f"ABUSE_PAY_{ring_id:03d}"
        )

        for customer_id in members:

            customer_rows = df[
                df["customer_id"]
                == customer_id
            ].index

            # Shared identifiers create
            # the graph relationship.
            df.loc[
                customer_rows,
                "device_id"
            ] = shared_device

            if rng.random() < 0.85:
                df.loc[
                    customer_rows,
                    "shipping_address_id"
                ] = shared_address

            if rng.random() < 0.65:
                df.loc[
                    customer_rows,
                    "payment_instrument_id"
                ] = shared_payment

            # Make behavior suspicious,
            # but not identical.
            for idx in customer_rows:

                if rng.random() < 0.65:
                    df.loc[
                        idx,
                        "is_returned"
                    ] = 1

                    df.loc[
                        idx,
                        "return_status"
                    ] = "Returned"

                    df.loc[
                        idx,
                        "days_to_return"
                    ] = int(
                        rng.integers(
                            1,
                            8,
                        )
                    )

                    df.loc[
                        idx,
                        "return_reason"
                    ] = rng.choice(
                        [
                            "Changed Mind",
                            "Wrong Item",
                            "Size Issue",
                            "Defective",
                        ]
                    )

                    df.loc[
                        idx,
                        "return_cost"
                    ] = float(
                        df.loc[
                            idx,
                            "order_value"
                        ]
                        * rng.uniform(
                            0.5,
                            0.9,
                        )
                    )

            df.loc[
                customer_rows,
                "is_abuse"
            ] = 1

            df.loc[
                customer_rows,
                "abuse_type"
            ] = "COORDINATED_RETURN_ABUSE"

            df.loc[
                customer_rows,
                "abuse_ring_id"
            ] = f"RING_{ring_id:03d}"

        cursor += ring_size
        ring_id += 1

    return df, ring_customers


# ---------------------------------------------------------
# ADD DEFAULT RING VALUE
# ---------------------------------------------------------

def initialize_ring_column(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df["abuse_ring_id"] = ""

    return df


# ---------------------------------------------------------
# VALIDATE
# ---------------------------------------------------------

def validate_labels(
    df: pd.DataFrame,
) -> None:

    customer_labels = (
        df.groupby("customer_id")
        .agg(
            is_abuse=("is_abuse", "max"),
            abuse_type=(
                "abuse_type",
                "first",
            ),
        )
        .reset_index()
    )

    print("\n--- LABEL SUMMARY ---")

    print(
        customer_labels[
            "abuse_type"
        ].value_counts()
    )

    print(
        "\nAbusive customers:",
        int(
            customer_labels[
                "is_abuse"
            ].sum()
        ),
    )

    print(
        "Legitimate customers:",
        int(
            (
                customer_labels[
                    "is_abuse"
                ] == 0
            ).sum()
        ),
    )

    print(
        "\nOrder-level abuse rate:",
        f"{df['is_abuse'].mean():.2%}",
    )

    ring_count = (
        df.loc[
            df["abuse_ring_id"] != "",
            "abuse_ring_id",
        ]
        .nunique()
    )

    print(
        "Abuse rings created:",
        ring_count,
    )

    print(
        "\nUnique devices after injection:",
        df["device_id"].nunique(),
    )

    print(
        "Unique addresses after injection:",
        df[
            "shipping_address_id"
        ].nunique(),
    )

    print(
        "Unique payment instruments after injection:",
        df[
            "payment_instrument_id"
        ].nunique(),
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    df = load_data()

    df = initialize_labels(df)

    df = initialize_ring_column(df)

    summary = build_customer_summary(df)

    df = inject_hard_negatives(
        df,
        summary,
    )

    # Rebuild summary after modifications
    summary = build_customer_summary(df)

    df, individual_abusers = (
        inject_individual_abuse(
            df,
            summary,
        )
    )

    excluded = set(
        individual_abusers
    )

    df, ring_customers = (
        inject_abuse_rings(
            df,
            summary,
            excluded,
        )
    )

    validate_labels(df)

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nSaved labeled dataset to:"
    )

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()