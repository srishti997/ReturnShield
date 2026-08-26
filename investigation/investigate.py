from pathlib import Path
from functools import lru_cache
import sys

import joblib
import pandas as pd
import shap


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =========================================================
# PATHS
# =========================================================

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_features.csv"
)

RINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "detected_rings.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "behavior_network_model.pkl"
)


# =========================================================
# LOAD + CACHE RESOURCES
# =========================================================

@lru_cache(maxsize=1)
def load_resources():

    print("Loading ReturnShield resources...")

    customers = pd.read_csv(
        FEATURES_PATH
    )

    rings = pd.read_csv(
        RINGS_PATH
    )

    model_package = joblib.load(
        MODEL_PATH
    )

    print(
        f"Loaded {len(customers)} customers, "
        f"{len(rings)} rings and ML model."
    )

    return (
        customers,
        rings,
        model_package,
    )


# =========================================================
# CACHE SHAP EXPLAINER
# =========================================================

@lru_cache(maxsize=1)
def get_shap_explainer():

    _, _, model_package = (
        load_resources()
    )

    model = model_package[
        "model"
    ]

    explainer = shap.TreeExplainer(
        model
    )

    return explainer


# =========================================================
# FIND CUSTOMER
# =========================================================

def get_customer(
    customers,
    customer_id,
):

    match = customers[
        customers[
            "customer_id"
        ] == customer_id
    ]

    if match.empty:

        raise ValueError(
            f"Customer {customer_id} not found."
        )

    return match.iloc[0]


# =========================================================
# ML RISK
# =========================================================

def calculate_ml_risk(
    customer,
    model_package,
):

    model = model_package[
        "model"
    ]

    features = model_package[
        "features"
    ]

    threshold = model_package[
        "threshold"
    ]

    X = pd.DataFrame(
        [
            customer[
                features
            ].to_dict()
        ]
    )

    probability = (
        model.predict_proba(
            X
        )[0][1]
    )

    flagged = (
        probability
        >= threshold
    )

    return (
        float(
            probability
        ),
        bool(
            flagged
        ),
        float(
            threshold
        ),
    )


# =========================================================
# SHAP EXPLANATION
# =========================================================

def calculate_shap_explanation(
    customer,
    model_package,
):

    features = model_package[
        "features"
    ]

    X = pd.DataFrame(
        [
            customer[
                features
            ].to_dict()
        ]
    )

    explainer = (
        get_shap_explainer()
    )

    shap_result = explainer(
        X
    )

    shap_values = (
        shap_result.values[0]
    )

    contributions = []

    for feature, shap_value in zip(
        features,
        shap_values,
    ):

        feature_value = (
            X.iloc[0][
                feature
            ]
        )

        contributions.append(
            {
                "feature":
                    feature,

                "feature_value":
                    float(
                        feature_value
                    ),

                "shap_value":
                    float(
                        shap_value
                    ),

                "direction":
                    (
                        "INCREASES_RISK"
                        if shap_value > 0
                        else "DECREASES_RISK"
                    ),
            }
        )

    contributions.sort(
        key=lambda item: abs(
            item[
                "shap_value"
            ]
        ),
        reverse=True,
    )

    return contributions[:5]


# =========================================================
# FIND CUSTOMER RING
# =========================================================

def find_customer_ring(
    rings,
    customer_id,
):

    for _, ring in rings.iterrows():

        members = (
            str(
                ring[
                    "members"
                ]
            )
            .split(",")
        )

        if customer_id in members:

            return ring

    return None


# =========================================================
# COMBINED RISK
# =========================================================

def get_risk_level(
    ml_probability,
    ring,
):

    graph_score = 0.0

    if ring is not None:

        graph_score = (
            float(
                ring[
                    "ring_risk_score"
                ]
            )
            / 100
        )

    # Prototype interpretable weighting.
    #
    # 70% ML model score
    # 30% graph cluster score

    combined_score = (
        0.70
        * ml_probability
        +
        0.30
        * graph_score
    )

    if combined_score >= 0.75:

        risk_level = "HIGH"

    elif combined_score >= 0.45:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    return (
        float(
            combined_score
        ),
        risk_level,
    )


# =========================================================
# BUILD EVIDENCE
# =========================================================

def build_evidence(
    customer,
    ring,
):

    evidence = []


    # -----------------------------------------------------
    # BEHAVIOURAL SIGNALS
    # -----------------------------------------------------

    return_rate = float(
        customer[
            "return_rate"
        ]
    )

    rapid_return_ratio = float(
        customer[
            "rapid_return_ratio"
        ]
    )

    return_cost_ratio = float(
        customer[
            "return_cost_ratio"
        ]
    )


    if return_rate >= 0.70:

        evidence.append(
            f"Very high return rate "
            f"({return_rate:.1%})"
        )

    elif return_rate >= 0.50:

        evidence.append(
            f"Elevated return rate "
            f"({return_rate:.1%})"
        )


    if rapid_return_ratio >= 0.50:

        evidence.append(
            f"High proportion of "
            f"rapid returns "
            f"({rapid_return_ratio:.1%})"
        )


    if return_cost_ratio >= 0.50:

        evidence.append(
            f"Large share of purchase "
            f"value returned "
            f"({return_cost_ratio:.1%})"
        )


    # -----------------------------------------------------
    # NETWORK SIGNALS
    # -----------------------------------------------------

    accounts_same_device = int(
        customer[
            "accounts_same_device"
        ]
    )

    accounts_same_address = int(
        customer[
            "accounts_same_address"
        ]
    )

    accounts_same_payment = int(
        customer[
            "accounts_same_payment"
        ]
    )


    if accounts_same_device > 1:

        evidence.append(
            f"Device shared across "
            f"{accounts_same_device} accounts"
        )


    if accounts_same_address > 1:

        evidence.append(
            f"Shipping address shared across "
            f"{accounts_same_address} accounts"
        )


    if accounts_same_payment > 1:

        evidence.append(
            f"Payment instrument shared across "
            f"{accounts_same_payment} accounts"
        )


    # -----------------------------------------------------
    # GRAPH SIGNALS
    # -----------------------------------------------------

    if ring is not None:

        member_count = int(
            ring[
                "member_count"
            ]
        )

        ring_score = float(
            ring[
                "ring_risk_score"
            ]
        )

        evidence.append(
            f"Member of a connected "
            f"{member_count}-account cluster"
        )

        evidence.append(
            f"Cluster risk score "
            f"{ring_score:.1f}/100"
        )

    return evidence


# =========================================================
# RECOMMEND ACTION
# =========================================================

def recommend_action(
    risk_level,
):

    if risk_level == "HIGH":

        return (
            "MANUAL REVIEW — verify the return "
            "before refund approval."
        )

    if risk_level == "MEDIUM":

        return (
            "STEP-UP VERIFICATION — request "
            "additional return evidence."
        )

    return (
        "ALLOW STANDARD FLOW — no elevated "
        "risk intervention recommended."
    )


# =========================================================
# MAIN INVESTIGATION
# =========================================================

def investigate(
    customer_id,
):

    (
        customers,
        rings,
        model_package,
    ) = load_resources()


    # -----------------------------------------------------
    # CUSTOMER
    # -----------------------------------------------------

    customer = get_customer(
        customers,
        customer_id,
    )


    # -----------------------------------------------------
    # ML
    # -----------------------------------------------------

    (
        ml_probability,
        model_flagged,
        model_threshold,
    ) = calculate_ml_risk(
        customer,
        model_package,
    )


    # -----------------------------------------------------
    # SHAP
    # -----------------------------------------------------

    shap_explanation = (
        calculate_shap_explanation(
            customer,
            model_package,
        )
    )


    # -----------------------------------------------------
    # GRAPH
    # -----------------------------------------------------

    ring = find_customer_ring(
        rings,
        customer_id,
    )


    # -----------------------------------------------------
    # FINAL RISK
    # -----------------------------------------------------

    (
        combined_score,
        risk_level,
    ) = get_risk_level(
        ml_probability,
        ring,
    )


    # -----------------------------------------------------
    # EVIDENCE
    # -----------------------------------------------------

    evidence = build_evidence(
        customer,
        ring,
    )


    # -----------------------------------------------------
    # ACTION
    # -----------------------------------------------------

    recommendation = (
        recommend_action(
            risk_level
        )
    )


    # =====================================================
    # RESPONSE
    # =====================================================

    result = {

        "customer_id":
            customer_id,

        "risk_level":
            risk_level,

        "combined_score":
            combined_score,

        "ml_probability":
            ml_probability,

        "model_threshold":
            model_threshold,

        "model_flagged":
            model_flagged,

        "return_rate":
            float(
                customer[
                    "return_rate"
                ]
            ),

        "rapid_return_ratio":
            float(
                customer[
                    "rapid_return_ratio"
                ]
            ),

        "return_cost_ratio":
            float(
                customer[
                    "return_cost_ratio"
                ]
            ),

        "return_value":
            float(
                customer[
                    "total_return_cost"
                ]
            ),

        "accounts_same_device":
            int(
                customer[
                    "accounts_same_device"
                ]
            ),

        "accounts_same_address":
            int(
                customer[
                    "accounts_same_address"
                ]
            ),

        "accounts_same_payment":
            int(
                customer[
                    "accounts_same_payment"
                ]
            ),

        "ring_id":
            (
                str(
                    ring[
                        "detected_ring_id"
                    ]
                )
                if ring is not None
                else None
            ),

        "ring_score":
            (
                float(
                    ring[
                        "ring_risk_score"
                    ]
                )
                if ring is not None
                else None
            ),

        "ring_member_count":
            (
                int(
                    ring[
                        "member_count"
                    ]
                )
                if ring is not None
                else 0
            ),

        "evidence":
            evidence,

        "shap_explanation":
            shap_explanation,

        "recommendation":
            recommendation,
    }

    return result


# =========================================================
# PRINT INVESTIGATION
# =========================================================

def print_investigation(
    result,
):

    print(
        "\n======================================"
    )

    print(
        "RETURNSHIELD CASE INVESTIGATION"
    )

    print(
        "======================================"
    )

    print(
        f"\nCustomer: "
        f"{result['customer_id']}"
    )

    print(
        f"Risk level: "
        f"{result['risk_level']}"
    )

    print(
        f"Combined risk score: "
        f"{result['combined_score'] * 100:.1f}/100"
    )

    print(
        f"Model risk score: "
        f"{result['ml_probability']:.1%}"
    )

    print(
        f"Model threshold: "
        f"{result['model_threshold']:.3f}"
    )

    print(
        f"Return rate: "
        f"{result['return_rate']:.1%}"
    )

    print(
        f"Rapid return ratio: "
        f"{result['rapid_return_ratio']:.1%}"
    )

    print(
        f"Return exposure: "
        f"₹{result['return_value']:,.2f}"
    )


    # -----------------------------------------------------
    # NETWORK
    # -----------------------------------------------------

    print(
        "\nNETWORK"
    )

    print(
        "Accounts sharing device:",
        result[
            "accounts_same_device"
        ],
    )

    print(
        "Accounts sharing address:",
        result[
            "accounts_same_address"
        ],
    )

    print(
        "Accounts sharing payment:",
        result[
            "accounts_same_payment"
        ],
    )


    if result[
        "ring_id"
    ] is not None:

        print(
            f"Connected cluster: "
            f"{result['ring_id']}"
        )

        print(
            f"Cluster members: "
            f"{result['ring_member_count']}"
        )

        print(
            f"Cluster risk: "
            f"{result['ring_score']:.1f}/100"
        )

    else:

        print(
            "Connected cluster: None"
        )


    # -----------------------------------------------------
    # EVIDENCE
    # -----------------------------------------------------

    print(
        "\nWHY THIS DECISION"
    )

    if result[
        "evidence"
    ]:

        for item in result[
            "evidence"
        ]:

            print(
                f"  - {item}"
            )

    else:

        print(
            "  - No strong risk "
            "indicators detected."
        )


    # -----------------------------------------------------
    # SHAP
    # -----------------------------------------------------

    print(
        "\nTOP ML RISK DRIVERS"
    )

    for item in result[
        "shap_explanation"
    ]:

        print(
            f"  - {item['feature']} | "
            f"value={item['feature_value']:.3f} | "
            f"SHAP={item['shap_value']:.3f} | "
            f"{item['direction']}"
        )


    # -----------------------------------------------------
    # ACTION
    # -----------------------------------------------------

    print(
        "\nRECOMMENDED ACTION"
    )

    print(
        result[
            "recommendation"
        ]
    )

    print(
        "\n======================================"
    )


# =========================================================
# COMMAND LINE
# =========================================================

def main():

    if len(
        sys.argv
    ) < 2:

        print(
            "\nUsage:"
        )

        print(
            "python investigation/"
            "investigate.py USER1403"
        )

        return

    customer_id = (
        sys.argv[1]
    )

    try:

        result = investigate(
            customer_id
        )

        print_investigation(
            result
        )

    except ValueError as error:

        print(
            f"\nError: {error}"
        )


if __name__ == "__main__":
    main()