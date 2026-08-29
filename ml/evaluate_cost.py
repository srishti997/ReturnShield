from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import (
    train_test_split,
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_features.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "ml"
    / "models"
)

MODEL_A_PATH = (
    MODEL_DIR
    / "behavior_model.pkl"
)

MODEL_B_PATH = (
    MODEL_DIR
    / "behavior_network_model.pkl"
)

COMPARISON_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_comparison.csv"
)

PREDICTIONS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_predictions.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Loaded dataset: "
        f"{len(df)} customers"
    )

    return df


# =========================================================
# RECREATE EXACT SAME TEST SPLIT
# =========================================================

def get_test_set(
    df,
):

    train_df, temp_df = (
        train_test_split(
            df,
            test_size=0.30,
            stratify=df[
                "is_abuse"
            ],
            random_state=42,
        )
    )

    val_df, test_df = (
        train_test_split(
            temp_df,
            test_size=0.50,
            stratify=temp_df[
                "is_abuse"
            ],
            random_state=42,
        )
    )

    print()
    print(
        "TEST SET"
    )
    print(
        "--------"
    )
    print(
        f"Customers: {len(test_df)}"
    )

    print(
        "Abusive customers:",
        int(
            test_df[
                "is_abuse"
            ].sum()
        ),
    )

    print(
        "Legitimate customers:",
        int(
            (
                test_df[
                    "is_abuse"
                ]
                == 0
            ).sum()
        ),
    )

    return test_df.copy()


# =========================================================
# LOAD SAVED MODEL PACKAGE
# =========================================================

def load_model(
    path,
):

    package = joblib.load(
        path
    )

    return (
        package[
            "model"
        ],
        package[
            "features"
        ],
        float(
            package[
                "threshold"
            ]
        ),
    )


# =========================================================
# GENERATE MODEL PREDICTIONS
# =========================================================

def generate_predictions(
    test_df,
    model,
    features,
    threshold,
):

    probabilities = (
        model.predict_proba(
            test_df[
                features
            ]
        )[:, 1]
    )

    predictions = (
        probabilities
        >= threshold
    ).astype(int)

    return (
        probabilities,
        predictions,
    )


# =========================================================
# CLASSIFICATION METRICS
# =========================================================

def calculate_classification_metrics(
    y_true,
    probabilities,
    predictions,
    threshold,
):

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
            labels=[
                0,
                1,
            ],
        )
        .ravel()
    )

    return {

        "threshold":
            threshold,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "pr_auc":
            pr_auc,

        "true_positives":
            int(tp),

        "false_positives":
            int(fp),

        "false_negatives":
            int(fn),

        "true_negatives":
            int(tn),
    }


# =========================================================
# MONETARY IMPACT
# =========================================================

def calculate_money_metrics(
    test_df,
    predictions,
):

    evaluation_df = (
        test_df.copy()
    )

    evaluation_df[
        "predicted_abuse"
    ] = predictions


    # -----------------------------------------------------
    # TRUE POSITIVES
    # -----------------------------------------------------

    true_positive = (
        evaluation_df[
            (
                evaluation_df[
                    "is_abuse"
                ] == 1
            )
            &
            (
                evaluation_df[
                    "predicted_abuse"
                ] == 1
            )
        ]
    )


    # -----------------------------------------------------
    # FALSE NEGATIVES
    # -----------------------------------------------------

    false_negative = (
        evaluation_df[
            (
                evaluation_df[
                    "is_abuse"
                ] == 1
            )
            &
            (
                evaluation_df[
                    "predicted_abuse"
                ] == 0
            )
        ]
    )


    # -----------------------------------------------------
    # FALSE POSITIVES
    # -----------------------------------------------------

    false_positive = (
        evaluation_df[
            (
                evaluation_df[
                    "is_abuse"
                ] == 0
            )
            &
            (
                evaluation_df[
                    "predicted_abuse"
                ] == 1
            )
        ]
    )


    # -----------------------------------------------------
    # VALUE CALCULATIONS
    # -----------------------------------------------------

    total_abusive_value = (
        evaluation_df.loc[
            evaluation_df[
                "is_abuse"
            ] == 1,
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
    # VALUE DETECTION RATE
    # -----------------------------------------------------

    if total_abusive_value > 0:

        value_detection_rate = (
            detected_abusive_value
            / total_abusive_value
        )

    else:

        value_detection_rate = 0.0


    return {

        "total_abusive_value":
            float(
                total_abusive_value
            ),

        "detected_abusive_value":
            float(
                detected_abusive_value
            ),

        "missed_abusive_value":
            float(
                missed_abusive_value
            ),

        "legitimate_flagged_value":
            float(
                legitimate_flagged_value
            ),

        "value_detection_rate":
            float(
                value_detection_rate
            ),
    }


# =========================================================
# EVALUATE ONE MODEL
# =========================================================

def evaluate_model(
    name,
    model_path,
    test_df,
):

    model, features, threshold = (
        load_model(
            model_path
        )
    )

    probabilities, predictions = (
        generate_predictions(
            test_df,
            model,
            features,
            threshold,
        )
    )

    classification_metrics = (
        calculate_classification_metrics(
            test_df[
                "is_abuse"
            ],
            probabilities,
            predictions,
            threshold,
        )
    )

    money_metrics = (
        calculate_money_metrics(
            test_df,
            predictions,
        )
    )

    metrics = {
        "model":
            name,
        **classification_metrics,
        **money_metrics,
    }

    return (
        metrics,
        probabilities,
        predictions,
    )


# =========================================================
# PRINT MODEL RESULTS
# =========================================================

def print_model_results(
    metrics,
):

    print()
    print(
        "======================================"
    )

    print(
        metrics[
            "model"
        ]
    )

    print(
        "======================================"
    )

    print(
        "Threshold:",
        f"{metrics['threshold']:.3f}",
    )

    print(
        "Precision:",
        f"{metrics['precision']:.3f}",
    )

    print(
        "Recall:",
        f"{metrics['recall']:.3f}",
    )

    print(
        "F1:",
        f"{metrics['f1']:.3f}",
    )

    print(
        "PR-AUC:",
        f"{metrics['pr_auc']:.3f}",
    )

    print()

    print(
        "CONFUSION MATRIX"
    )

    print(
        "----------------"
    )

    print(
        "True positives:",
        metrics[
            "true_positives"
        ],
    )

    print(
        "False positives:",
        metrics[
            "false_positives"
        ],
    )

    print(
        "False negatives:",
        metrics[
            "false_negatives"
        ],
    )

    print(
        "True negatives:",
        metrics[
            "true_negatives"
        ],
    )

    print()

    print(
        "FINANCIAL IMPACT"
    )

    print(
        "----------------"
    )

    print(
        "Total abusive return value:",
        f"₹{metrics['total_abusive_value']:,.2f}",
    )

    print(
        "Detected abusive value:",
        f"₹{metrics['detected_abusive_value']:,.2f}",
    )

    print(
        "Missed abusive value:",
        f"₹{metrics['missed_abusive_value']:,.2f}",
    )

    print(
        "Legitimate value flagged:",
        f"₹{metrics['legitimate_flagged_value']:,.2f}",
    )

    print(
        "Value detection rate:",
        f"{metrics['value_detection_rate']:.2%}",
    )


# =========================================================
# PRINT FINAL COMPARISON
# =========================================================

def print_comparison(
    model_a,
    model_b,
):

    print()
    print()
    print(
        "=========================================="
    )

    print(
        "RETURNSHIELD MODEL COMPARISON"
    )

    print(
        "=========================================="
    )

    comparison = pd.DataFrame(
        [
            {
                "Metric":
                    "Precision",

                "Behavior Only":
                    model_a[
                        "precision"
                    ],

                "Behavior + Network":
                    model_b[
                        "precision"
                    ],
            },

            {
                "Metric":
                    "Recall",

                "Behavior Only":
                    model_a[
                        "recall"
                    ],

                "Behavior + Network":
                    model_b[
                        "recall"
                    ],
            },

            {
                "Metric":
                    "F1",

                "Behavior Only":
                    model_a[
                        "f1"
                    ],

                "Behavior + Network":
                    model_b[
                        "f1"
                    ],
            },

            {
                "Metric":
                    "PR-AUC",

                "Behavior Only":
                    model_a[
                        "pr_auc"
                    ],

                "Behavior + Network":
                    model_b[
                        "pr_auc"
                    ],
            },

            {
                "Metric":
                    "False Positives",

                "Behavior Only":
                    model_a[
                        "false_positives"
                    ],

                "Behavior + Network":
                    model_b[
                        "false_positives"
                    ],
            },

            {
                "Metric":
                    "False Negatives",

                "Behavior Only":
                    model_a[
                        "false_negatives"
                    ],

                "Behavior + Network":
                    model_b[
                        "false_negatives"
                    ],
            },

            {
                "Metric":
                    "Value Detection Rate",

                "Behavior Only":
                    model_a[
                        "value_detection_rate"
                    ],

                "Behavior + Network":
                    model_b[
                        "value_detection_rate"
                    ],
            },

            {
                "Metric":
                    "Detected Abuse Value",

                "Behavior Only":
                    model_a[
                        "detected_abusive_value"
                    ],

                "Behavior + Network":
                    model_b[
                        "detected_abusive_value"
                    ],
            },

            {
                "Metric":
                    "Missed Abuse Value",

                "Behavior Only":
                    model_a[
                        "missed_abusive_value"
                    ],

                "Behavior + Network":
                    model_b[
                        "missed_abusive_value"
                    ],
            },
        ]
    )

    print()

    print(
        comparison.to_string(
            index=False
        )
    )

    return comparison


# =========================================================
# PRINT IMPROVEMENT SUMMARY
# =========================================================

def print_improvement_summary(
    model_a,
    model_b,
):

    recall_improvement = (
        model_b[
            "recall"
        ]
        - model_a[
            "recall"
        ]
    )

    f1_improvement = (
        model_b[
            "f1"
        ]
        - model_a[
            "f1"
        ]
    )

    pr_auc_improvement = (
        model_b[
            "pr_auc"
        ]
        - model_a[
            "pr_auc"
        ]
    )

    additional_value_detected = (
        model_b[
            "detected_abusive_value"
        ]
        - model_a[
            "detected_abusive_value"
        ]
    )

    fewer_false_negatives = (
        model_a[
            "false_negatives"
        ]
        - model_b[
            "false_negatives"
        ]
    )


    print()
    print(
        "=========================================="
    )

    print(
        "NETWORK INTELLIGENCE IMPACT"
    )

    print(
        "=========================================="
    )

    print(
        "Recall improvement:",
        f"{recall_improvement:+.3f}",
    )

    print(
        "F1 improvement:",
        f"{f1_improvement:+.3f}",
    )

    print(
        "PR-AUC improvement:",
        f"{pr_auc_improvement:+.3f}",
    )

    print(
        "Fewer abusive customers missed:",
        fewer_false_negatives,
    )

    print(
        "Additional abusive value detected:",
        f"₹{additional_value_detected:,.2f}",
    )


# =========================================================
# EXPORT COMPARISON
# =========================================================

def export_comparison(
    model_a,
    model_b,
):

    comparison_df = pd.DataFrame(
        [
            model_a,
            model_b,
        ]
    )

    comparison_df.to_csv(
        COMPARISON_OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "Saved model comparison to:"
    )

    print(
        COMPARISON_OUTPUT_PATH
    )


# =========================================================
# EXPORT MODEL B TEST PREDICTIONS
# =========================================================

def export_model_b_predictions(
    test_df,
    probabilities,
    predictions,
):

    output_df = (
        test_df.copy()
    )

    output_df[
        "risk_probability"
    ] = probabilities

    output_df[
        "predicted_abuse"
    ] = predictions

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

    output_df[
        columns
    ].sort_values(
        "risk_probability",
        ascending=False,
    ).to_csv(
        PREDICTIONS_OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "Saved Model B predictions to:"
    )

    print(
        PREDICTIONS_OUTPUT_PATH
    )


# =========================================================
# MAIN
# =========================================================

def main():

    df = load_data()

    test_df = get_test_set(
        df
    )


    # =====================================================
    # MODEL A
    # =====================================================

    (
        model_a_metrics,
        model_a_probabilities,
        model_a_predictions,
    ) = evaluate_model(
        "MODEL A - BEHAVIOR",
        MODEL_A_PATH,
        test_df,
    )


    # =====================================================
    # MODEL B
    # =====================================================

    (
        model_b_metrics,
        model_b_probabilities,
        model_b_predictions,
    ) = evaluate_model(
        "MODEL B - BEHAVIOR + NETWORK",
        MODEL_B_PATH,
        test_df,
    )


    # =====================================================
    # DISPLAY INDIVIDUAL RESULTS
    # =====================================================

    print_model_results(
        model_a_metrics
    )

    print_model_results(
        model_b_metrics
    )


    # =====================================================
    # COMPARISON
    # =====================================================

    print_comparison(
        model_a_metrics,
        model_b_metrics,
    )

    print_improvement_summary(
        model_a_metrics,
        model_b_metrics,
    )


    # =====================================================
    # EXPORT FILES
    # =====================================================

    export_comparison(
        model_a_metrics,
        model_b_metrics,
    )

    export_model_b_predictions(
        test_df,
        model_b_probabilities,
        model_b_predictions,
    )


if __name__ == "__main__":

    main()