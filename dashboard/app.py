from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ReturnShield",
    page_icon="🛡️",
    layout="wide",
)


# =========================================================
# CONFIG
# =========================================================

API_BASE_URL = "http://127.0.0.1:8000"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GRAPH_OUTPUT_DIR = PROJECT_ROOT / "graph" / "outputs"


# =========================================================
# API HELPERS
# =========================================================

def api_get(endpoint, params=None):

    try:

        response = requests.get(
            f"{API_BASE_URL}{endpoint}",
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as error:

        st.error(
            "Could not connect to ReturnShield API.\n\n"
            f"{error}"
        )

        return None


def api_post(endpoint, payload):

    try:

        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=payload,
            timeout=30,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as error:

        st.error(
            "Could not connect to ReturnShield API.\n\n"
            f"{error}"
        )

        return None


# =========================================================
# HEADER
# =========================================================

st.title("🛡️ ReturnShield")

st.caption(
    "AI-powered return-abuse detection and "
    "coordinated account-risk investigation"
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "Risk Overview",
        "Customer Investigation",
        "Live Return Evaluation",
        "Abuse Rings",
        "Model Performance",
    ],
)


# =========================================================
# PAGE 1 — RISK OVERVIEW
# =========================================================

if page == "Risk Overview":

    st.header(
        "Merchant Risk Overview"
    )

    metrics = api_get(
        "/metrics"
    )

    queue = api_get(
        "/risk-queue",
        params={
            "limit": 20,
            "risk_level": "HIGH",
        },
    )

    rings = api_get(
        "/rings",
        params={
            "limit": 50,
        },
    )

    if metrics and queue and rings:

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        col1.metric(
            "High-risk cases shown",
            queue[
                "cases_returned"
            ],
        )

        exposure = queue[
            "total_return_exposure"
        ]

        col2.metric(
            "Risk exposure",
            f"₹{exposure:,.0f}",
        )

        col3.metric(
            "Detected abuse rings",
            rings[
                "count"
            ],
        )

        detection_rate = metrics[
            "value_detection_rate"
        ]

        col4.metric(
            "Value detection",
            f"{detection_rate:.1%}",
        )

        st.divider()

        st.subheader(
            "High-Risk Investigation Queue"
        )

        queue_df = pd.DataFrame(
            queue[
                "cases"
            ]
        )

        if not queue_df.empty:

            display_df = queue_df[
                [
                    "customer_id",
                    "risk_score",
                    "return_rate",
                    "return_exposure",
                    "ring_id",
                ]
            ].copy()

            display_df.columns = [
                "Customer",
                "Risk Score",
                "Return Rate %",
                "Exposure ₹",
                "Cluster",
            ]

            display_df[
                "Risk Score"
            ] = (
                display_df[
                    "Risk Score"
                ]
                .round(2)
            )

            display_df[
                "Return Rate %"
            ] = (
                display_df[
                    "Return Rate %"
                ]
                .round(2)
            )

            display_df[
                "Exposure ₹"
            ] = (
                display_df[
                    "Exposure ₹"
                ]
                .round(2)
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "No high-risk cases found."
            )

        st.caption(
            "Cases are ranked by ReturnShield risk score."
        )


# =========================================================
# PAGE 2 — CUSTOMER INVESTIGATION
# =========================================================

elif page == "Customer Investigation":

    st.header(
        "Customer Investigation"
    )

    st.caption(
        "Investigate an existing customer using "
        "behavioural ML signals and account-network relationships."
    )

    customer_id = st.text_input(
        "Customer ID",
        value="USER1403",
        placeholder="Example: USER1403",
    )

    analyze_button = st.button(
        "Analyze Customer",
        type="primary",
    )

    if analyze_button:

        result = api_get(
            f"/customers/"
            f"{customer_id}/"
            f"investigation"
        )

        if result:

            st.divider()

            risk_level = result[
                "risk_level"
            ]

            combined_score = (
                result[
                    "combined_score"
                ]
                * 100
            )

            model_score = result[
                "ml_probability"
            ]

            return_value = result[
                "return_value"
            ]

            ring_score = result[
                "ring_score"
            ]

            recommendation = result[
                "recommendation"
            ]

            st.subheader(
                f"Investigation: "
                f"{result['customer_id']}"
            )

            if risk_level == "HIGH":

                st.error(
                    f"🔴 HIGH RISK — "
                    f"ReturnShield Score: "
                    f"{combined_score:.1f}/100"
                )

            elif risk_level == "MEDIUM":

                st.warning(
                    f"🟡 MEDIUM RISK — "
                    f"ReturnShield Score: "
                    f"{combined_score:.1f}/100"
                )

            else:

                st.success(
                    f"🟢 LOW RISK — "
                    f"ReturnShield Score: "
                    f"{combined_score:.1f}/100"
                )

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            col1.metric(
                "ReturnShield Score",
                f"{combined_score:.1f}/100",
            )

            col2.metric(
                "ML Abuse Probability",
                f"{model_score:.1%}",
            )

            col3.metric(
                "Return Exposure",
                f"₹{return_value:,.2f}",
            )

            if ring_score is not None:

                ring_display = (
                    f"{ring_score:.1f}/100"
                )

            else:

                ring_display = (
                    "No cluster"
                )

            col4.metric(
                "Cluster Risk",
                ring_display,
            )

            st.divider()

            behaviour_col, network_col = (
                st.columns(2)
            )

            with behaviour_col:

                st.subheader(
                    "📊 Behaviour Intelligence"
                )

                return_rate = result[
                    "return_rate"
                ]

                rapid_ratio = result[
                    "rapid_return_ratio"
                ]

                return_cost_ratio = result[
                    "return_cost_ratio"
                ]

                b1, b2 = st.columns(2)

                b1.metric(
                    "Return Rate",
                    f"{return_rate:.1%}",
                )

                b2.metric(
                    "Rapid Return Ratio",
                    f"{rapid_ratio:.1%}",
                )

                st.metric(
                    "Returned Value Ratio",
                    f"{return_cost_ratio:.1%}",
                )

                st.caption(
                    "Behavioural signals are evaluated "
                    "using the trained XGBoost model."
                )

            with network_col:

                st.subheader(
                    "🕸️ Network Intelligence"
                )

                n1, n2 = st.columns(2)

                n1.metric(
                    "Accounts / Device",
                    result[
                        "accounts_same_device"
                    ],
                )

                n2.metric(
                    "Accounts / Address",
                    result[
                        "accounts_same_address"
                    ],
                )

                n3, n4 = st.columns(2)

                n3.metric(
                    "Accounts / Payment",
                    result[
                        "accounts_same_payment"
                    ],
                )

                n4.metric(
                    "Cluster Members",
                    result[
                        "ring_member_count"
                    ],
                )

                if result[
                    "ring_id"
                ]:

                    st.warning(
                        "Connected suspicious cluster: "
                        f"{result['ring_id']}"
                    )

                else:

                    st.success(
                        "No suspicious account cluster detected."
                    )

            st.divider()

            st.header(
                "Why This Case Was Flagged"
            )

            evidence_col, graph_col = (
                st.columns(
                    [1, 1.4]
                )
            )

            with evidence_col:

                st.subheader(
                    "Risk Evidence"
                )

                evidence = result.get(
                    "evidence",
                    [],
                )

                if evidence:

                    for number, item in enumerate(
                        evidence,
                        start=1,
                    ):

                        st.write(
                            f"{number}. {item}"
                        )

                else:

                    st.info(
                        "No additional evidence available."
                    )

            with graph_col:

                st.subheader(
                    "Account Relationship Graph"
                )

                ring_id = result[
                    "ring_id"
                ]

                if ring_id:

                    graph_path = (
                        GRAPH_OUTPUT_DIR
                        / f"{ring_id}.html"
                    )

                    if graph_path.exists():

                        graph_html = (
                            graph_path.read_text(
                                encoding="utf-8"
                            )
                        )

                        components.html(
                            graph_html,
                            height=520,
                            scrolling=False,
                        )

                    else:

                        st.info(
                            "Graph visualization has not "
                            "been generated for this cluster."
                        )

                else:

                    st.info(
                        "No relationship graph is associated "
                        "with this customer."
                    )

            st.divider()

            st.header(
                "Recommended Action"
            )

            if risk_level == "HIGH":

                st.error(
                    f"⚠️ {recommendation}"
                )

            elif risk_level == "MEDIUM":

                st.warning(
                    f"⚠️ {recommendation}"
                )

            else:

                st.success(
                    f"✅ {recommendation}"
                )

            st.caption(
                "ReturnShield provides a risk recommendation. "
                "The merchant remains responsible for the final decision."
            )


# =========================================================
# PAGE 3 — LIVE RETURN EVALUATION
# =========================================================

elif page == "Live Return Evaluation":

    st.header(
        "Live Return Evaluation"
    )

    st.caption(
        "Evaluate a new return request in real time using "
        "historical behaviour, network intelligence and "
        "current return context."
    )

    st.info(
        "This simulates the point at which a merchant "
        "receives a new return request."
    )

    left, right = (
        st.columns(2)
    )

    with left:

        live_customer_id = st.text_input(
            "Customer ID",
            value="USER1403",
            key="live_customer_id",
        )

        order_value = st.number_input(
            "Return / Order Value (₹)",
            min_value=0.0,
            value=4500.0,
            step=100.0,
        )

    with right:

        return_reason = st.selectbox(
            "Return Reason",
            [
                "Changed Mind",
                "Defective",
                "Wrong Item",
                "Size Issue",
                "Other",
            ],
        )

        days_to_return = st.number_input(
            "Days Since Purchase",
            min_value=0,
            max_value=365,
            value=2,
            step=1,
        )

    evaluate_button = st.button(
        "Evaluate Return",
        type="primary",
    )

    if evaluate_button:

        payload = {

            "customer_id":
                live_customer_id,

            "order_value":
                float(
                    order_value
                ),

            "return_reason":
                return_reason,

            "days_to_return":
                int(
                    days_to_return
                ),
        }

        result = api_post(
            "/evaluate-return",
            payload,
        )

        if result:

            st.divider()

            risk_level = result[
                "risk_level"
            ]

            final_score = result[
                "final_return_risk_score"
            ]

            historical_score = result[
                "historical_risk_score"
            ]

            model_risk_score = result.get(
                "model_risk_score"
            )

            customer_status = result[
                "customer_status"
            ]

            decision_basis = result.get(
                "decision_basis",
                "UNKNOWN",
            )

            # =================================================
            # SAFE DISPLAY VALUES FOR NEW CUSTOMERS
            # =================================================

            if historical_score is None:

                historical_display = (
                    "Unavailable"
                )

            else:

                historical_display = (
                    f"{historical_score:.1f}/100"
                )

            if model_risk_score is None:

                model_display = (
                    "Unavailable"
                )

            else:

                model_display = (
                    f"{model_risk_score:.1f}/100"
                )

            st.subheader(
                f"Return Decision: "
                f"{result['customer_id']}"
            )

            # =================================================
            # CUSTOMER TYPE
            # =================================================

            if customer_status == "NEW":

                st.info(
                    "🆕 New customer — no historical "
                    "ML behaviour is available. "
                    "ReturnShield is using cold-start "
                    "current-return evaluation."
                )

            else:

                st.info(
                    "Existing customer — historical behaviour "
                    "and network intelligence are available."
                )

            # =================================================
            # RISK RESULT
            # =================================================

            if risk_level == "HIGH":

                st.error(
                    f"🔴 HIGH RISK — "
                    f"{final_score:.1f}/100"
                )

            elif risk_level == "MEDIUM":

                st.warning(
                    f"🟡 MEDIUM RISK — "
                    f"{final_score:.1f}/100"
                )

            else:

                st.success(
                    f"🟢 LOW RISK — "
                    f"{final_score:.1f}/100"
                )

            # =================================================
            # TOP METRICS
            # =================================================

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            col1.metric(
                "Final Return Risk",
                f"{final_score:.1f}/100",
            )

            col2.metric(
                "Historical Risk",
                historical_display,
            )

            col3.metric(
                "Return Value",
                f"₹{result['order_value']:,.2f}",
            )

            col4.metric(
                "Days to Return",
                result[
                    "days_to_return"
                ],
            )

            # =================================================
            # EXTRA MODEL INFORMATION
            # =================================================

            st.divider()

            model_col, status_col, basis_col = (
                st.columns(3)
            )

            model_col.metric(
                "Historical ML Score",
                model_display,
            )

            status_col.metric(
                "Customer Status",
                customer_status,
            )

            if decision_basis == (
                "HISTORICAL_MODEL_PLUS_CURRENT_CONTEXT"
            ):

                basis_display = (
                    "History + Current Return"
                )

            elif decision_basis == (
                "COLD_START_CURRENT_CONTEXT"
            ):

                basis_display = (
                    "Cold Start"
                )

            else:

                basis_display = (
                    decision_basis
                )

            basis_col.metric(
                "Decision Basis",
                basis_display,
            )

            st.divider()

            # =================================================
            # RETURN + NETWORK CONTEXT
            # =================================================

            context_col, network_col = (
                st.columns(2)
            )

            with context_col:

                st.subheader(
                    "📦 Current Return Context"
                )

                st.write(
                    "Customer:",
                    result[
                        "customer_id"
                    ],
                )

                st.write(
                    "Customer status:",
                    customer_status,
                )

                st.write(
                    "Return value:",
                    f"₹{result['order_value']:,.2f}",
                )

                st.write(
                    "Reason:",
                    result[
                        "return_reason"
                    ],
                )

                st.write(
                    "Days since purchase:",
                    result[
                        "days_to_return"
                    ],
                )

            with network_col:

                st.subheader(
                    "🕸️ Network Context"
                )

                ring_id = result[
                    "ring_id"
                ]

                if ring_id:

                    st.warning(
                        "Customer belongs to suspicious "
                        f"cluster {ring_id}."
                    )

                elif customer_status == "NEW":

                    st.info(
                        "No known account-network history "
                        "is available for this new customer."
                    )

                else:

                    st.success(
                        "No suspicious connected cluster found."
                    )

            st.divider()

            # =================================================
            # EVIDENCE
            # =================================================

            st.subheader(
                "Why ReturnShield Made This Decision"
            )

            evidence = result.get(
                "evidence",
                [],
            )

            if evidence:

                for number, item in enumerate(
                    evidence,
                    start=1,
                ):

                    st.write(
                        f"{number}. {item}"
                    )

            else:

                st.info(
                    "No elevated risk indicators detected."
                )

            st.divider()

            # =================================================
            # RECOMMENDATION
            # =================================================

            st.subheader(
                "Recommended Decision"
            )

            action = result[
                "recommended_action"
            ]

            if risk_level == "HIGH":

                st.error(
                    f"⚠️ {action}"
                )

            elif risk_level == "MEDIUM":

                st.warning(
                    f"⚠️ {action}"
                )

            else:

                st.success(
                    f"✅ {action}"
                )

            st.caption(
                "ReturnShield provides a risk recommendation "
                "rather than automatically rejecting a return."
            )

            st.divider()

            # =================================================
            # DECISION SUMMARY
            # =================================================

            decision_df = pd.DataFrame(
                {

                    "Factor": [
                        "Customer status",
                        "Historical risk",
                        "Historical ML score",
                        "Current return value",
                        "Return timing",
                        "Return reason",
                        "Connected ring",
                        "Final ReturnShield score",
                        "Decision basis",
                        "Recommended action",
                    ],

                    "Result": [
                        customer_status,
                        historical_display,
                        model_display,
                        f"₹{result['order_value']:,.2f}",
                        (
                            f"{result['days_to_return']} days"
                        ),
                        result[
                            "return_reason"
                        ],
                        (
                            result[
                                "ring_id"
                            ]
                            if result[
                                "ring_id"
                            ]
                            else "None"
                        ),
                        f"{final_score:.1f}/100",
                        basis_display,
                        action,
                    ],
                }
            )

            st.dataframe(
                decision_df,
                use_container_width=True,
                hide_index=True,
            )


# =========================================================
# PAGE 4 — ABUSE RINGS
# =========================================================

elif page == "Abuse Rings":

    st.header(
        "Detected Account Clusters"
    )

    st.caption(
        "Groups of customer accounts connected through "
        "shared devices, addresses or payment instruments."
    )

    rings = api_get(
        "/rings",
        params={
            "limit": 50,
        },
    )

    if rings:

        ring_df = pd.DataFrame(
            rings[
                "rings"
            ]
        )

        if not ring_df.empty:

            display_df = ring_df[
                [
                    "ring_id",
                    "risk_score",
                    "member_count",
                    "return_rate",
                    "return_value",
                    "device_links",
                    "address_links",
                    "payment_links",
                ]
            ].copy()

            display_df.columns = [
                "Cluster",
                "Risk Score",
                "Members",
                "Avg Return Rate %",
                "Return Exposure ₹",
                "Device Links",
                "Address Links",
                "Payment Links",
            ]

            display_df[
                "Avg Return Rate %"
            ] = (
                display_df[
                    "Avg Return Rate %"
                ]
                * 100
            ).round(1)

            display_df[
                "Risk Score"
            ] = (
                display_df[
                    "Risk Score"
                ]
                .round(2)
            )

            display_df[
                "Return Exposure ₹"
            ] = (
                display_df[
                    "Return Exposure ₹"
                ]
                .round(2)
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "No connected account clusters found."
            )


# =========================================================
# PAGE 5 — MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.header(
        "Held-Out Model Evaluation"
    )

    metrics = api_get(
        "/metrics"
    )

    if metrics:

        st.caption(
            metrics[
                "note"
            ]
        )

        precision = metrics[
            "precision"
        ]

        recall = metrics[
            "recall"
        ]

        f1_score = metrics[
            "f1"
        ]

        pr_auc = metrics[
            "pr_auc"
        ]

        m1, m2, m3, m4 = (
            st.columns(4)
        )

        m1.metric(
            "Precision",
            f"{precision:.1%}",
        )

        m2.metric(
            "Recall",
            f"{recall:.1%}",
        )

        m3.metric(
            "F1 Score",
            f"{f1_score:.1%}",
        )

        m4.metric(
            "PR-AUC",
            f"{pr_auc:.3f}",
        )

        st.divider()

        st.subheader(
            "Financial Impact"
        )

        total_abusive_value = metrics[
            "total_abusive_return_value"
        ]

        detected_value = metrics[
            "detected_abusive_value"
        ]

        missed_value = metrics[
            "missed_abusive_value"
        ]

        value_detection_rate = metrics[
            "value_detection_rate"
        ]

        f1_col, f2_col, f3_col, f4_col = (
            st.columns(4)
        )

        f1_col.metric(
            "Abusive Value",
            f"₹{total_abusive_value:,.2f}",
        )

        f2_col.metric(
            "Value Detected",
            f"₹{detected_value:,.2f}",
        )

        f3_col.metric(
            "Value Missed",
            f"₹{missed_value:,.2f}",
        )

        f4_col.metric(
            "Value Detection Rate",
            f"{value_detection_rate:.1%}",
        )

        st.divider()

        st.subheader(
            "Test Set Outcomes"
        )

        confusion_df = pd.DataFrame(
            {

                "Outcome": [
                    "True Positive",
                    "False Positive",
                    "False Negative",
                    "True Negative",
                ],

                "Count": [
                    metrics[
                        "true_positives"
                    ],
                    metrics[
                        "false_positives"
                    ],
                    metrics[
                        "false_negatives"
                    ],
                    metrics[
                        "true_negatives"
                    ],
                ],
            }
        )

        st.dataframe(
            confusion_df,
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "These metrics come from the held-out "
            "synthetic-augmented prototype test set "
            "and should not be interpreted as "
            "production performance."
        )