from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "returns_sustainability_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_PATH = OUTPUT_DIR / "returnshield_base.csv"


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


# ---------------------------------------------------------
# CLEAN DATA
# ---------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    # Columns that are not relevant to ReturnShield
    columns_to_drop = [
        "User_Age",
        "User_Gender",
        "CO2_Emissions",
        "Packaging_Waste",
        "CO2_Saved",
        "Waste_Avoided",
    ]

    df = df.drop(columns=columns_to_drop)

    # Rename columns into consistent snake_case
    df = df.rename(
        columns={
            "Order_ID": "order_id",
            "Product_ID": "product_id",
            "User_ID": "customer_id",
            "Order_Date": "order_date",
            "Product_Category": "product_category",
            "Product_Price": "product_price",
            "Order_Quantity": "order_quantity",
            "Discount_Applied": "discount_applied",
            "Shipping_Method": "shipping_method",
            "Payment_Method": "payment_method",
            "User_Location": "user_location",
            "Return_Status": "return_status",
            "Return_Reason": "return_reason",
            "Days_to_Return": "days_to_return",
            "Order_Value": "order_value",
            "Return_Cost": "return_cost",
            "Profit_Loss": "profit_loss",
        }
    )

    # Convert order date to datetime
    df["order_date"] = pd.to_datetime(df["order_date"])

    # Binary indicator for return
    df["is_returned"] = (
        df["return_status"]
        .str.lower()
        .eq("returned")
        .astype(int)
    )

    return df


# ---------------------------------------------------------
# CREATE CUSTOMER-LEVEL IDENTITIES
# ---------------------------------------------------------

def create_identity_mapping(df: pd.DataFrame) -> pd.DataFrame:

    customers = (
        df[["customer_id", "user_location"]]
        .drop_duplicates("customer_id")
        .copy()
    )

    n_customers = len(customers)

    # One primary device for most customers
    customers["device_id"] = [
        f"DEV_{i:05d}"
        for i in range(1, n_customers + 1)
    ]

    # One primary shipping address
    customers["shipping_address_id"] = [
        f"ADDR_{i:05d}"
        for i in range(1, n_customers + 1)
    ]

    # One payment instrument
    customers["payment_instrument_id"] = [
        f"PAY_{i:05d}"
        for i in range(1, n_customers + 1)
    ]

    # Create synthetic IPs
    customers["ip_address"] = [
        (
            f"10."
            f"{rng.integers(1, 255)}."
            f"{rng.integers(1, 255)}."
            f"{rng.integers(1, 255)}"
        )
        for _ in range(n_customers)
    ]

    return customers


# ---------------------------------------------------------
# CREATE ACCOUNT AGE
# ---------------------------------------------------------

def add_account_information(
    df: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:

    first_order = (
        df.groupby("customer_id")["order_date"]
        .min()
        .rename("first_order_date")
    )

    customers = customers.merge(
        first_order,
        on="customer_id",
        how="left",
    )

    # Assume account was created between
    # 1 and 900 days before first purchase.
    days_before_first_order = rng.integers(
        1,
        901,
        size=len(customers),
    )

    customers["account_created_date"] = (
        customers["first_order_date"]
        - pd.to_timedelta(days_before_first_order, unit="D")
    )

    customers = customers.drop(columns=["first_order_date"])

    return customers


# ---------------------------------------------------------
# MERGE IDENTITIES INTO ORDERS
# ---------------------------------------------------------

def merge_identity_data(
    df: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:

    identity_columns = [
        "customer_id",
        "device_id",
        "shipping_address_id",
        "payment_instrument_id",
        "ip_address",
        "account_created_date",
    ]

    df = df.merge(
        customers[identity_columns],
        on="customer_id",
        how="left",
    )

    df["account_age_days"] = (
        df["order_date"]
        - df["account_created_date"]
    ).dt.days

    return df


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> None:

    print("\n--- DATASET VALIDATION ---")

    print(f"Rows: {len(df)}")
    print(f"Customers: {df['customer_id'].nunique()}")
    print(f"Orders: {df['order_id'].nunique()}")

    print(
        f"Returned orders: "
        f"{df['is_returned'].sum()}"
    )

    print(
        f"Return rate: "
        f"{df['is_returned'].mean():.2%}"
    )

    print(
        f"Unique devices: "
        f"{df['device_id'].nunique()}"
    )

    print(
        f"Unique addresses: "
        f"{df['shipping_address_id'].nunique()}"
    )

    print(
        f"Unique payment instruments: "
        f"{df['payment_instrument_id'].nunique()}"
    )

    print("\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    df = clean_data(df)

    customers = create_identity_mapping(df)

    customers = add_account_information(
        df,
        customers,
    )

    df = merge_identity_data(
        df,
        customers,
    )

    validate_dataset(df)

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved prepared dataset to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()