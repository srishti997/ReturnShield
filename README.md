# 🛡️ ReturnShield

### AI-Powered Return Abuse Detection for Safer Refund Decisions

ReturnShield is an AI-powered risk intelligence system designed to help merchants identify potentially abusive e-commerce return requests **before a refund is approved**.

Instead of relying only on individual customer behaviour, ReturnShield combines:

- Behavioral machine learning
- Account-network intelligence
- Coordinated abuse-ring detection
- SHAP-based model explainability
- Persistent live customer history
- Return-level contextual risk
- Merchant-facing investigation tools

The system produces an explainable risk decision that can help merchants decide whether to:

**Allow the return → Request additional verification → Send for manual review**

> **Prototype disclosure:** ReturnShield is evaluated using a synthetic-augmented dataset created for prototype experimentation. Reported metrics should not be interpreted as production performance.

---

## 🎯 The Problem

Return fraud is not always obvious.

A customer with an unusually high return rate may be suspicious, but sophisticated abuse can involve multiple accounts that individually appear relatively normal.

For example:

```text
Customer A ─┐
Customer B ─┼── Same Device
Customer C ─┘

Customer A ───── Same Address ───── Customer D

Customer B ───── Same Payment ───── Customer E
```

Looking at these customers independently can miss the relationship.

ReturnShield therefore asks two questions:

> **Is this customer's behaviour suspicious?**

and

> **Is this customer connected to a suspicious network of accounts?**

---

# 💡 Solution

ReturnShield evaluates return risk using multiple layers of intelligence.

```text
                     RETURN REQUEST
                           │
                           ▼
                  ┌─────────────────┐
                  │ ReturnShield API │
                  └────────┬────────┘
                           │
            ┌──────────────┼───────────────┐
            │              │               │
            ▼              ▼               ▼
      Behavioral ML    Network Risk    Live History
        (XGBoost)       Intelligence     & Context
            │              │               │
            └──────────────┼───────────────┘
                           │
                           ▼
                    Risk Decision
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
             LOW         MEDIUM        HIGH
              │            │            │
              ▼            ▼            ▼
           Standard      Step-up       Manual
             Flow       Verification    Review
```

---

# 🔍 Core Capabilities

## 1. Behavioral Risk Detection

ReturnShield builds customer-level behavioural features such as:

- Return rate
- Rapid-return ratio
- Return-cost ratio
- Total order activity
- Average order value
- Return exposure
- Account age
- Purchase frequency
- Product/category diversity

An **XGBoost classifier** estimates the probability that a customer's behaviour resembles abusive return activity.

---

## 2. Network Intelligence

Behavior alone may miss coordinated abuse.

ReturnShield therefore incorporates relationships between accounts through shared identifiers:

- Device
- Shipping address
- Payment instrument

These relationships become additional ML features:

```text
accounts_same_device
accounts_same_address
accounts_same_payment
```

This allows the model to identify suspicious customers whose individual behaviour may not be sufficient to detect them.

---

## 3. Coordinated Abuse-Ring Detection

ReturnShield also constructs an account-identity graph.

```text
Customer
   │
   ├── Device
   │
   ├── Address
   │
   └── Payment Instrument
```

Customers sharing identities become connected through the graph.

The prototype detected:

```text
18 connected account groups
74 customers participating in connected groups
```

Each detected ring receives a risk score based on factors including:

- Shared identities
- Return behaviour
- Network density
- Return exposure

Interactive graph visualizations are generated for every detected ring.

Example:

```text
graph/outputs/DETECTED_010.html
```

---

## 4. Explainable AI with SHAP

A fraud score without an explanation is difficult for an investigator to act on.

ReturnShield uses **SHAP (SHapley Additive exPlanations)** to expose which ML features contributed most strongly to an individual prediction.

An investigation can therefore show:

```text
Risk Score
+
Behavioral Evidence
+
Network Evidence
+
Top ML Risk Drivers
+
Recommended Action
```

This helps an investigator understand **why** a customer was considered risky instead of treating the model as a black box.

---

## 5. Live Customer Memory

ReturnShield supports customers who were not present in the original model dataset.

For a completely new customer:

```text
NEW_CUSTOMER
        │
        ▼
Cold-start evaluation
        │
        ▼
Return event stored
```

When the same customer returns later:

```text
RETURNING_LIVE_CUSTOMER
        │
        ▼
Previous return history retrieved
        │
        ▼
Historical behaviour
        +
Current return context
        │
        ▼
Updated risk decision
```

Live customer activity is persisted using SQLite.

This means ReturnShield can gradually accumulate behavioural intelligence for previously unseen customers.

---

# 🧠 Risk Decision Paths

ReturnShield supports three customer states.

### Existing Historical Customer

```text
EXISTING_DATASET_CUSTOMER

Behavioral ML
+
Network Intelligence
+
Historical Evidence
+
Current Return Context
```

### Returning Live Customer

```text
RETURNING_LIVE_CUSTOMER

Stored Live History
+
Previous Return Frequency
+
Previous Return Exposure
+
Rapid Return Behaviour
+
Current Return Context
```

### New Customer

```text
NEW_CUSTOMER

Cold-Start Rules
+
Current Return Context
```

---

# 📊 Model Evaluation

Two models were evaluated on the same held-out test set.

### Model A

Behavioral features only.

### Model B

Behavioral + network features.

Test set:

```text
279 customers
25 abusive
254 legitimate
```

| Metric | Behavior Only | Behavior + Network |
|---|---:|---:|
| Precision | 1.000 | **1.000** |
| Recall | 0.800 | **0.920** |
| F1 Score | 0.889 | **0.958** |
| PR-AUC | 0.842 | **0.938** |
| False Positives | 0 | **0** |
| False Negatives | 5 | **2** |

### Network Intelligence Impact

Adding network features:

- Increased recall from **80% → 92%**
- Increased F1 from **88.9% → 95.8%**
- Increased PR-AUC from **84.2% → 93.8%**
- Reduced missed abusive customers from **5 → 2**
- Introduced **no additional false positives** on this test set

These results are measured on the held-out synthetic-augmented prototype test set.

---

# 💰 Return-Value Impact

ReturnShield also evaluates risk in monetary terms.

For the Behavior + Network model:

| Metric | Result |
|---|---:|
| Total abusive return value | ₹74,569.63 |
| Abusive value detected | **₹73,987.63** |
| Abusive value missed | ₹582.00 |
| Legitimate value incorrectly flagged | ₹0.00 |
| Value detection rate | **99.22%** |

Compared with behavioral ML alone, network intelligence identified an additional:

```text
₹400.00
```

of abusive return value on the test set.

---

# 🔗 Example Abuse Ring

One detected cluster is:

```text
DETECTED_010
```

It contains five connected customer accounts.

The group exhibits shared:

- Devices
- Shipping addresses
- Payment instruments

with a calculated ring risk score of approximately:

```text
93.67 / 100
```

The interactive graph allows investigators to visually inspect how these accounts are connected.

---

# 🕵️ Example Investigation

A high-risk prototype case:

```text
Customer: USER1403
Risk Level: HIGH
Combined Risk Score: ~98.1 / 100
ML Abuse Probability: ~100%
Return Rate: 100%
Return Exposure: ~₹1,484.51

Ring:
DETECTED_010

Ring Risk:
~93.7 / 100
```

ReturnShield surfaces behavioral, network and model evidence before recommending:

```text
MANUAL REVIEW
```

---

# 🖥️ Merchant Dashboard

ReturnShield includes a Streamlit investigation dashboard with five primary views.

### Risk Overview

Displays:

- High-risk cases
- Return exposure
- Detected rings
- Value detection rate
- Merchant risk queue

### Customer Investigation

Allows an investigator to inspect:

- Risk score
- Behavioral metrics
- Network relationships
- Abuse-ring membership
- Evidence
- SHAP model drivers
- Recommended action

### Live Return Evaluation

Allows a merchant to evaluate a return request using:

```text
Customer ID
Order Value
Return Reason
Days to Return
```

### Abuse Rings

Provides visibility into suspicious connected account groups.

### Model Performance

Displays model-quality and financial-impact metrics.

---

# ⚙️ API

ReturnShield exposes a FastAPI service.

### Health

```http
GET /health
```

### Customers

```http
GET /customers
```

### Customer Risk

```http
GET /customers/{customer_id}/risk
```

### Full Investigation

```http
GET /customers/{customer_id}/investigation
```

### Abuse Rings

```http
GET /rings
GET /rings/{ring_id}
```

### Risk Queue

```http
GET /risk-queue
```

### Model Metrics

```http
GET /metrics
```

### Live Return Evaluation

```http
POST /evaluate-return
```

Example request:

```json
{
  "customer_id": "CUSTOMER_001",
  "order_value": 6500,
  "return_reason": "Changed Mind",
  "days_to_return": 1
}
```

Example decision:

```json
{
  "customer_status": "RETURNING_LIVE_CUSTOMER",
  "final_return_risk_score": 65,
  "risk_level": "MEDIUM",
  "recommended_action": "STEP-UP VERIFICATION"
}
```

---

# 🏗️ Technology Stack

| Layer | Technology |
|---|---|
| Machine Learning | XGBoost |
| ML Evaluation | scikit-learn |
| Explainability | SHAP |
| Graph Analysis | NetworkX |
| Graph Visualization | PyVis |
| API | FastAPI |
| Dashboard | Streamlit |
| Data Processing | Pandas / NumPy |
| Persistence | SQLite |
| Model Serialization | Joblib |
| Language | Python |

---

# 📁 Project Structure

```text
returnshield/
│
├── api/
│   └── main.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── live/
│
├── graph/
│   ├── build_graph.py
│   ├── detect_rings.py
│   ├── visualize.py
│   └── outputs/
│
├── investigation/
│   └── investigate.py
│
├── ml/
│   ├── features.py
│   ├── train.py
│   ├── evaluate_cost.py
│   └── models/
│
├── storage/
│   └── database.py
│
├── requirements.txt
└── README.md
```

---

# 🚀 Running ReturnShield

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Start the API

From the project root:

```bash
uvicorn api.main:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## 3. Start the dashboard

Open another terminal:

```bash
streamlit run dashboard/app.py
```

Dashboard:

```text
http://localhost:8501
```

---

# 🧪 Reproducing Model Evaluation

Run:

```bash
python ml/evaluate_cost.py
```

This evaluates both:

```text
Behavior Only
vs.
Behavior + Network
```

on the same held-out test set and generates:

```text
data/processed/model_comparison.csv
data/processed/test_predictions.csv
```

---

# 🕸️ Generate Abuse-Ring Visualizations

Run:

```bash
python graph/visualize.py
```

ReturnShield generates an interactive HTML visualization for every detected ring under:

```text
graph/outputs/
```

---

# ⚠️ Prototype Limitations

ReturnShield is a prototype rather than a production fraud-decision system.

Important limitations include:

- The dataset has been synthetically augmented to simulate coordinated abuse relationships.
- Model results therefore should not be generalized to real-world production traffic.
- Device, address and payment relationships are simplified representations.
- Cold-start risk currently uses deterministic contextual rules.
- SQLite is used for lightweight prototype persistence.
- Production deployment would require stronger identity resolution, streaming infrastructure, monitoring and model governance.
- High-risk predictions should support human investigation rather than automatically denying legitimate customer returns.

---

# 🔮 Production Evolution

A production implementation could extend ReturnShield with:

```text
Merchant Return Events
        │
        ▼
Streaming Event Pipeline
        │
        ├──── Feature Store
        │
        ├──── Graph Intelligence
        │
        └──── Historical Behaviour
                     │
                     ▼
               Risk Service
                     │
             ┌───────┼────────┐
             ▼       ▼        ▼
            LOW    MEDIUM     HIGH
             │       │         │
          Approve  Verify    Review
```

Potential improvements include:

- Real-time feature stores
- Streaming behavioral updates
- Graph databases
- Device fingerprinting
- Identity-resolution models
- Temporal graph analysis
- Merchant-specific thresholds
- Drift monitoring
- Feedback from investigator decisions
- Periodic model retraining

---

# 🛡️ ReturnShield

**Behavior tells us what one account is doing.  
Networks help reveal who may be acting together.**

ReturnShield combines both to make return-risk decisions more explainable, contextual and useful for merchant investigation.