from pathlib import Path
import sys

import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT)
)


from investigation.investigate import investigate


# =========================================================
# DATA PATHS
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

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_predictions.csv"
)


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="ReturnShield API",
    description=(
        "AI-powered return-abuse detection, "
        "network-risk analysis and explainable "
        "merchant return investigation."
    ),
    version="1.0.0",
)


# =========================================================
# REQUEST MODELS
# =========================================================

class ReturnRequest(BaseModel):
    customer_id: str
    order_value: float
    return_reason: str
    days_to_return: int


# =========================================================
# LOAD DATA
# =========================================================

customers_df = pd.read_csv(
    FEATURES_PATH
)

rings_df = pd.read_csv(
    RINGS_PATH
)

predictions_df = pd.read_csv(
    PREDICTIONS_PATH
)


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "service": "ReturnShield",
        "status": "running",
        "version": "1.0.0",
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "customers_loaded": len(
            customers_df
        ),
        "rings_loaded": len(
            rings_df
        ),
        "test_predictions_loaded": len(
            predictions_df
        ),
    }


# =========================================================
# CUSTOMER LIST
# =========================================================

@app.get("/customers")
def list_customers(
    limit: int = 20,
):

    limit = min(
        max(limit, 1),
        100,
    )

    customers = (
        customers_df[
            [
                "customer_id",
                "return_rate",
                "total_return_cost",
                "accounts_same_device",
                "accounts_same_address",
                "accounts_same_payment",
            ]
        ]
        .head(limit)
        .to_dict(
            orient="records"
        )
    )

    return {
        "count": len(
            customers
        ),
        "customers": customers,
    }


# =========================================================
# CUSTOMER BASIC RISK FEATURES
# =========================================================

@app.get(
    "/customers/{customer_id}/risk"
)
def customer_risk(
    customer_id: str,
):

    match = customers_df[
        customers_df[
            "customer_id"
        ] == customer_id
    ]

    if match.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Customer "
                f"{customer_id} not found."
            ),
        )

    customer = match.iloc[0]

    return {

        "customer_id":
            customer_id,

        "return_rate":
            float(
                customer[
                    "return_rate"
                ]
            ),

        "return_value":
            float(
                customer[
                    "total_return_cost"
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
    }


# =========================================================
# FULL CUSTOMER INVESTIGATION
# =========================================================

@app.get(
    "/customers/{customer_id}/investigation"
)
def customer_investigation(
    customer_id: str,
):

    try:

        result = investigate(
            customer_id
        )

        return result

    except ValueError as error:

        raise HTTPException(
            status_code=404,
            detail=str(
                error
            ),
        )


# =========================================================
# LIST DETECTED RINGS
# =========================================================

@app.get("/rings")
def list_rings(
    limit: int = 20,
):

    limit = min(
        max(limit, 1),
        100,
    )

    rings = (
        rings_df
        .sort_values(
            "ring_risk_score",
            ascending=False,
        )
        .head(limit)
    )

    records = []

    for _, ring in rings.iterrows():

        records.append(
            {

                "ring_id":
                    ring[
                        "detected_ring_id"
                    ],

                "member_count":
                    int(
                        ring[
                            "member_count"
                        ]
                    ),

                "members":
                    str(
                        ring[
                            "members"
                        ]
                    ).split(","),

                "risk_score":
                    float(
                        ring[
                            "ring_risk_score"
                        ]
                    ),

                "return_rate":
                    float(
                        ring[
                            "avg_return_rate"
                        ]
                    ),

                "return_value":
                    float(
                        ring[
                            "total_return_value"
                        ]
                    ),

                "network_density":
                    float(
                        ring[
                            "network_density"
                        ]
                    ),

                "device_links":
                    int(
                        ring[
                            "device_links"
                        ]
                    ),

                "address_links":
                    int(
                        ring[
                            "address_links"
                        ]
                    ),

                "payment_links":
                    int(
                        ring[
                            "payment_links"
                        ]
                    ),
            }
        )

    return {
        "count": len(
            records
        ),
        "rings": records,
    }


# =========================================================
# SINGLE RING
# =========================================================

@app.get(
    "/rings/{ring_id}"
)
def get_ring(
    ring_id: str,
):

    match = rings_df[
        rings_df[
            "detected_ring_id"
        ] == ring_id
    ]

    if match.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Ring "
                f"{ring_id} not found."
            ),
        )

    ring = match.iloc[0]

    return {

        "ring_id":
            ring_id,

        "member_count":
            int(
                ring[
                    "member_count"
                ]
            ),

        "members":
            str(
                ring[
                    "members"
                ]
            ).split(","),

        "risk_score":
            float(
                ring[
                    "ring_risk_score"
                ]
            ),

        "network_density":
            float(
                ring[
                    "network_density"
                ]
            ),

        "average_return_rate":
            float(
                ring[
                    "avg_return_rate"
                ]
            ),

        "return_value":
            float(
                ring[
                    "total_return_value"
                ]
            ),

        "device_links":
            int(
                ring[
                    "device_links"
                ]
            ),

        "address_links":
            int(
                ring[
                    "address_links"
                ]
            ),

        "payment_links":
            int(
                ring[
                    "payment_links"
                ]
            ),
    }


# =========================================================
# MODEL PERFORMANCE
# =========================================================

@app.get("/metrics")
def metrics():

    return {

        "test_customers":
            279,

        "precision":
            1.0,

        "recall":
            0.92,

        "f1":
            0.9583,

        "pr_auc":
            0.9377,

        "true_positives":
            23,

        "false_positives":
            0,

        "false_negatives":
            2,

        "true_negatives":
            254,

        "total_abusive_return_value":
            74569.63,

        "detected_abusive_value":
            73987.63,

        "missed_abusive_value":
            582.00,

        "legitimate_flagged_value":
            0.00,

        "value_detection_rate":
            0.9922,

        "note": (
            "Metrics are measured on the "
            "held-out synthetic-augmented "
            "prototype test set."
        ),
    }


# =========================================================
# MERCHANT RISK QUEUE
# =========================================================

@app.get("/risk-queue")
def risk_queue(
    limit: int = 20,
    risk_level: str = "HIGH",
):

    limit = min(
        max(limit, 1),
        100,
    )

    risk_level = (
        risk_level
        .strip()
        .upper()
    )

    allowed_levels = {
        "HIGH",
        "MEDIUM",
        "LOW",
        "ALL",
    }

    if risk_level not in allowed_levels:

        raise HTTPException(
            status_code=400,
            detail=(
                "risk_level must be "
                "HIGH, MEDIUM, LOW or ALL."
            ),
        )

    results = []

    for customer_id in customers_df[
        "customer_id"
    ].unique():

        try:

            result = investigate(
                customer_id
            )

        except ValueError:

            continue

        if (
            risk_level != "ALL"
            and result[
                "risk_level"
            ] != risk_level
        ):
            continue

        results.append(
            {

                "customer_id":
                    customer_id,

                "risk_level":
                    result[
                        "risk_level"
                    ],

                "risk_score":
                    round(
                        result[
                            "combined_score"
                        ] * 100,
                        2,
                    ),

                "model_score":
                    round(
                        result[
                            "ml_probability"
                        ] * 100,
                        2,
                    ),

                "return_rate":
                    round(
                        result[
                            "return_rate"
                        ] * 100,
                        2,
                    ),

                "return_exposure":
                    round(
                        result[
                            "return_value"
                        ],
                        2,
                    ),

                "ring_id":
                    result[
                        "ring_id"
                    ],

                "ring_score":
                    (
                        round(
                            result[
                                "ring_score"
                            ],
                            2,
                        )
                        if result[
                            "ring_score"
                        ] is not None
                        else None
                    ),

                "recommended_action":
                    result[
                        "recommendation"
                    ],
            }
        )

    results.sort(
        key=lambda item: (
            item[
                "risk_score"
            ],
            item[
                "return_exposure"
            ],
        ),
        reverse=True,
    )

    results = results[
        :limit
    ]

    total_exposure = sum(
        item[
            "return_exposure"
        ]
        for item in results
    )

    return {

        "risk_level_filter":
            risk_level,

        "cases_returned":
            len(
                results
            ),

        "total_return_exposure":
            round(
                total_exposure,
                2,
            ),

        "cases":
            results,
    }


# =========================================================
# LIVE RETURN EVALUATION
# =========================================================

@app.post("/evaluate-return")
def evaluate_return(
    request: ReturnRequest,
):

    # -----------------------------------------------------
    # BASIC INPUT VALIDATION
    # -----------------------------------------------------

    if request.order_value < 0:

        raise HTTPException(
            status_code=400,
            detail=(
                "order_value cannot "
                "be negative."
            ),
        )

    if request.days_to_return < 0:

        raise HTTPException(
            status_code=400,
            detail=(
                "days_to_return cannot "
                "be negative."
            ),
        )

    customer_id = (
        request.customer_id
        .strip()
    )

    if not customer_id:

        raise HTTPException(
            status_code=400,
            detail=(
                "customer_id cannot "
                "be empty."
            ),
        )


    # =====================================================
    # CHECK WHETHER CUSTOMER EXISTS
    # =====================================================

    try:

        investigation = investigate(
            customer_id
        )

        customer_known = True

    except ValueError:

        investigation = None

        customer_known = False


    # =====================================================
    # BASE RISK
    # =====================================================

    evidence = []

    if customer_known:

        customer_status = (
            "EXISTING"
        )

        score = investigation[
            "combined_score"
        ]

        evidence.extend(
            investigation[
                "evidence"
            ]
        )

        historical_risk_score = round(
            investigation[
                "combined_score"
            ] * 100,
            2,
        )

        model_risk_score = round(
            investigation[
                "ml_probability"
            ] * 100,
            2,
        )

        ring_id = investigation[
            "ring_id"
        ]

        ring_score = investigation[
            "ring_score"
        ]

    else:

        # -------------------------------------------------
        # COLD-START CUSTOMER
        # -------------------------------------------------

        customer_status = (
            "NEW"
        )

        score = 0.0

        historical_risk_score = (
            None
        )

        model_risk_score = (
            None
        )

        ring_id = None

        ring_score = None

        evidence.append(
            "New customer — no historical "
            "behaviour available"
        )

        evidence.append(
            "No historical ML risk score "
            "available"
        )

        evidence.append(
            "No known account-network "
            "history available"
        )


    # =====================================================
    # CURRENT RETURN — TIMING
    # =====================================================

    if request.days_to_return <= 1:

        score += 0.25

        evidence.append(
            "Return requested within "
            "1 day of purchase"
        )

    elif request.days_to_return <= 3:

        score += 0.15

        evidence.append(
            f"Return requested only "
            f"{request.days_to_return} "
            f"days after purchase"
        )

    elif request.days_to_return <= 7:

        score += 0.05

        evidence.append(
            f"Return requested within "
            f"{request.days_to_return} "
            f"days"
        )


    # =====================================================
    # CURRENT RETURN — VALUE
    # =====================================================

    if request.order_value >= 15000:

        score += 0.30

        evidence.append(
            f"Very high-value return "
            f"request "
            f"(₹{request.order_value:,.2f})"
        )

    elif request.order_value >= 10000:

        score += 0.25

        evidence.append(
            f"High-value return request "
            f"(₹{request.order_value:,.2f})"
        )

    elif request.order_value >= 5000:

        score += 0.12

        evidence.append(
            f"Elevated-value return "
            f"request "
            f"(₹{request.order_value:,.2f})"
        )

    elif request.order_value >= 2500:

        score += 0.05

        evidence.append(
            f"Moderate-value return "
            f"request "
            f"(₹{request.order_value:,.2f})"
        )


    # =====================================================
    # RETURN REASON
    # =====================================================

    normalized_reason = (
        request.return_reason
        .strip()
        .lower()
    )

    if normalized_reason == (
        "changed mind"
    ):

        score += 0.08

        evidence.append(
            "Return reason is "
            "'Changed Mind'"
        )

    elif normalized_reason == (
        "defective"
    ):

        evidence.append(
            "Return reason is "
            "'Defective'"
        )

    elif normalized_reason == (
        "wrong item"
    ):

        evidence.append(
            "Return reason is "
            "'Wrong Item'"
        )

    elif normalized_reason == (
        "size issue"
    ):

        evidence.append(
            "Return reason is "
            "'Size Issue'"
        )

    else:

        evidence.append(
            "Other return reason "
            "provided"
        )


    # =====================================================
    # KEEP SCORE WITHIN 0–1
    # =====================================================

    score = min(
        max(
            score,
            0.0,
        ),
        1.0,
    )


    # =====================================================
    # FINAL RISK DECISION
    # =====================================================

    if score >= 0.75:

        risk_level = (
            "HIGH"
        )

        recommendation = (
            "MANUAL REVIEW — verify "
            "the return before refund "
            "approval."
        )

    elif score >= 0.45:

        risk_level = (
            "MEDIUM"
        )

        recommendation = (
            "STEP-UP VERIFICATION — "
            "request additional return "
            "evidence."
        )

    else:

        risk_level = (
            "LOW"
        )

        recommendation = (
            "ALLOW STANDARD FLOW — "
            "no elevated risk "
            "intervention recommended."
        )


    # =====================================================
    # FINAL RESPONSE
    # =====================================================

    return {

        "customer_id":
            customer_id,

        "customer_status":
            customer_status,

        "order_value":
            request.order_value,

        "return_reason":
            request.return_reason,

        "days_to_return":
            request.days_to_return,

        "historical_risk_score":
            historical_risk_score,

        "model_risk_score":
            model_risk_score,

        "final_return_risk_score":
            round(
                score * 100,
                2,
            ),

        "risk_level":
            risk_level,

        "ring_id":
            ring_id,

        "ring_score":
            ring_score,

        "evidence":
            evidence,

        "recommended_action":
            recommendation,

        "decision_basis":
            (
                "HISTORICAL_MODEL_PLUS_CURRENT_CONTEXT"
                if customer_known
                else "COLD_START_CURRENT_CONTEXT"
            ),
    }