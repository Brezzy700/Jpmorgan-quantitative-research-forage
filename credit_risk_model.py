import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# -------------------------------------------------------
# STEP 1: Load and explore the data
# -------------------------------------------------------
df = pd.read_csv("Task_3_and_4_Loan_Data.csv")

print("Shape:", df.shape)
print("\nDefault rate: {:.1%}".format(df["default"].mean()))
print("\nFirst few rows:")
print(df.head(3))

# -------------------------------------------------------
# STEP 2: Prepare features
# -------------------------------------------------------
features = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score"
]

X = df[features]
y = df["default"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# -------------------------------------------------------
# STEP 3: Train Logistic Regression model
# -------------------------------------------------------
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(classification_report(y_test, y_pred))
print("ROC-AUC Score: {:.4f}".format(roc_auc_score(y_test, y_prob)))

# -------------------------------------------------------
# STEP 4: Expected Loss Function
# -------------------------------------------------------
RECOVERY_RATE = 0.10  # 10% recovery rate as stated in task

def expected_loss(
    credit_lines_outstanding,
    loan_amt_outstanding,
    total_debt_outstanding,
    income,
    years_employed,
    fico_score
):
    """
    Estimates the expected loss on a loan given borrower details.

    Expected Loss = PD x LGD x EAD
    - PD  = Probability of Default (from logistic regression model)
    - LGD = Loss Given Default = 1 - recovery_rate = 90%
    - EAD = Exposure at Default = loan_amt_outstanding

    Parameters:
        credit_lines_outstanding : int
        loan_amt_outstanding     : float
        total_debt_outstanding   : float
        income                   : float
        years_employed           : int
        fico_score               : int

    Returns:
        dict with PD, LGD, EAD, and Expected Loss
    """
    input_data = pd.DataFrame([{
        "credit_lines_outstanding": credit_lines_outstanding,
        "loan_amt_outstanding":     loan_amt_outstanding,
        "total_debt_outstanding":   total_debt_outstanding,
        "income":                   income,
        "years_employed":           years_employed,
        "fico_score":               fico_score
    }])

    input_scaled = scaler.transform(input_data)
    pd_value = model.predict_proba(input_scaled)[0][1]  # Probability of Default
    lgd      = 1 - RECOVERY_RATE                         # Loss Given Default = 90%
    ead      = loan_amt_outstanding                       # Exposure at Default

    loss = pd_value * lgd * ead

    print("="*45)
    print("  EXPECTED LOSS CALCULATION")
    print("="*45)
    print(f"  Probability of Default (PD) : {pd_value:.4f} ({pd_value:.1%})")
    print(f"  Loss Given Default (LGD)    : {lgd:.0%}")
    print(f"  Exposure at Default (EAD)   : ${ead:,.2f}")
    print(f"  Expected Loss               : ${loss:,.2f}")
    print("="*45)

    return {
        "probability_of_default": round(pd_value, 4),
        "loss_given_default":     round(lgd, 2),
        "exposure_at_default":    round(ead, 2),
        "expected_loss":          round(loss, 2)
    }

# -------------------------------------------------------
# STEP 5: Test with sample borrowers
# -------------------------------------------------------
print("\nTEST 1 — High risk borrower (many credit lines, low FICO):")
result1 = expected_loss(
    credit_lines_outstanding=5,
    loan_amt_outstanding=5000,
    total_debt_outstanding=25000,
    income=40000,
    years_employed=1,
    fico_score=530
)

print("\nTEST 2 — Low risk borrower (no credit lines, high FICO):")
result2 = expected_loss(
    credit_lines_outstanding=0,
    loan_amt_outstanding=3000,
    total_debt_outstanding=2000,
    income=90000,
    years_employed=7,
    fico_score=750
)

print("\nTEST 3 — Mid-range borrower:")
result3 = expected_loss(
    credit_lines_outstanding=2,
    loan_amt_outstanding=4000,
    total_debt_outstanding=9000,
    income=65000,
    years_employed=4,
    fico_score=630
)
