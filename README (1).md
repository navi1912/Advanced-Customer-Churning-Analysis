# Customer Behaviour Analytics & Churn Forecasting

An end-to-end portfolio project that predicts customer churn from historical e-commerce behaviour using a **SQL-first / feature-engineering-first analytics workflow**, followed by **XGBoost classification** and **SHAP explainability**.

## Project Objective

The goal is to identify customers who are likely to become inactive so that a business can target them with retention actions such as reactivation offers, service recovery, or personalized recommendations.

For this prototype:

- **Prediction cutoff:** 30 September 2025
- **Churn definition:** a customer is considered churned if they make **no successful purchase during the next 90 days**
- The 90-day future window is used **only to create the target label**
- All model features are calculated using data available on or before the prediction cutoff to avoid **temporal data leakage**

## Dataset

This project uses a **synthetic e-commerce dataset** generated for portfolio and learning purposes.

The relational dataset contains six tables:

1. `customers` — customer profile and signup information
2. `transactions` — transaction-level purchase history
3. `transaction_items` — product-level details within transactions
4. `products` — product category and pricing information
5. `payments` — payment method and payment status
6. `support_interactions` — customer support history and satisfaction scores

The dataset contains more than **100K transaction records** and approximately **8K customers**.

> The data are synthetic and should not be interpreted as real company or client data. Model performance in this project demonstrates the pipeline, not expected production performance.

## Project Workflow

```text
Six relational tables
        ↓
Data-quality validation / EDA
        ↓
Prediction cutoff
        ↓
90-day churn label
        ↓
Historical successful transactions
        ↓
Customer-level feature engineering
        ↓
28 behavioural features
        ↓
Logistic Regression baseline
        ↓
XGBoost classifier
        ↓
ROC-AUC / Precision / Recall / F1
        ↓
SHAP explainability
        ↓
Customer-level retention actions
```

## Feature Engineering

The transaction-level data are transformed into one row per customer.

The project creates **28 behavioural features**, including:

### RFM and lifetime behaviour
- Recency
- Purchase frequency
- Monetary value
- Average order value
- Order-value variability
- Customer tenure
- Active customer span

### Purchase timing
- Average inter-purchase gap
- Standard deviation of purchase gaps
- Days since first purchase

### Recent activity
- Purchases in the last 30 days
- Purchases in the last 90 days
- Spend in the last 30 days
- Spend in the last 90 days
- Recent average order value

### Behavioural trends
- Recent vs previous 90-day purchase trend
- Recent vs previous 90-day spend trend
- Rolling order-value behaviour

### Engagement and operational features
- Mobile-channel share
- Product-category diversity
- Failed-payment rate
- Refund rate
- Recent support-contact count
- Average satisfaction score
- Cohort-retention metric

## Data Leakage Prevention

A strict temporal cutoff is used.

```text
Historical data                Future label window
---------------------------|-----------------------------
                           |
                    30 Sep 2025
```

All features are created using information available **on or before 30 Sep 2025**.

The following 90 days are used only to determine whether the customer churned.

The variable used to construct the target, `future_purchases_90d`, is never included as a predictive feature.

## Exploratory Data Analysis

The EDA pipeline includes:

- Schema and table-grain inspection
- Primary-key validation
- Duplicate checks
- Missing-value checks
- Transaction-date validation
- Payment-status distribution
- Customer-segment distribution
- Churn-class distribution
- Comparison of behavioural metrics between churners and non-churners

## Modeling

### Baseline

A **Logistic Regression** model is trained first to provide a simple baseline.

### XGBoost

An `XGBClassifier` is then trained because churn is a structured tabular classification problem and customer behaviour may contain nonlinear relationships and feature interactions.

Key parameters explored include:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `reg_lambda`

## Evaluation

The model is evaluated using:

- ROC-AUC
- Precision
- Recall
- F1-score
- Accuracy
- Confusion matrix

Accuracy is not used alone because churn classification can be affected by class imbalance.

## Explainability with SHAP

SHAP is used to explain:

- **Global feature importance** — which behavioural variables influence the model most overall
- **Local explanations** — why a specific customer received a high or low churn-risk score

This allows model predictions to be converted into business-readable explanations.

## Business Use

Predicted churn probabilities can be translated into customer-retention actions.

Examples:

- High recency / no recent purchases → reactivation campaign
- Repeated support issues + low satisfaction → service recovery
- Falling spend → personalized recommendation or loyalty offer
- High-risk customer with strong historical value → priority retention outreach

## Tech Stack

- Python
- Pandas
- NumPy
- PostgreSQL / SQL concepts
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Google Colab

## Repository Structure

```text
customer_churn_project/
│
├── data/
│   ├── customers.csv
│   ├── transactions.csv
│   ├── transaction_items.csv
│   ├── products.csv
│   ├── payments.csv
│   └── support_interactions.csv
│
├── sql/
│   ├── 01_schema.sql
│   └── 02_load_data.sql
│
├── data_dictionary.csv
├── Customer_Churn_Complete_Colab.ipynb
└── README.md
```

## How to Run

### Google Colab

1. Upload `Customer_Churn_Complete_Colab.ipynb` to Google Colab.
2. Upload the project data folder or ZIP file.
3. Extract it if required:

```python
!unzip -q customer_churn_project.zip
```

4. Run the notebook from top to bottom.

### Required Python packages

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib
```

## Important Note

This is a **portfolio prototype built on synthetic data**. It is intended to demonstrate an end-to-end churn analytics workflow, feature engineering, classification, explainability, and business interpretation.

For a production deployment, the model should be retrained and validated on real historical customer data using stronger temporal validation across multiple historical cutoffs.
