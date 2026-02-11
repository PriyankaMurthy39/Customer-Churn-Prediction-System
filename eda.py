import pandas as pd
import numpy as np

# ===============================
# 1. Load dataset
# ===============================
df = pd.read_csv("Telco-Customer-Churn.csv")

print("Dataset loaded")
print(df.head())
print(df.shape)

# ===============================
# 2. Basic info
# ===============================
print(df.info())
print(df.describe())

# ===============================
# 3. Fix TotalCharges column
# ===============================
# Convert to numeric (some values are spaces)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing TotalCharges with median
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# ===============================
# 4. Drop customerID (not useful for ML)
# ===============================
df.drop("customerID", axis=1, inplace=True)

# ===============================
# 5. Convert target variable
# ===============================
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# ===============================
# 6. Encode binary categorical columns
# ===============================
binary_cols = [
    "gender", "Partner", "Dependents",
    "PhoneService", "PaperlessBilling"
]

for col in binary_cols:
    df[col] = df[col].map({"No": 0, "Yes": 1, "Female": 0, "Male": 1})

# ===============================
# 7. One-hot encode remaining categorical columns
# ===============================
df = pd.get_dummies(
    df,
    columns=[
        "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ],
    drop_first=True
)

# ===============================
# 8. BUSINESS FEATURE ENGINEERING
# ===============================

# 8.1 Tenure groups
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-1 year", "1-2 years", "2-4 years", "4+ years"]
)

df = pd.get_dummies(df, columns=["TenureGroup"], drop_first=True)

# 8.2 High monthly charge flag
df["HighMonthlyCharge"] = (
    df["MonthlyCharges"] > df["MonthlyCharges"].median()
).astype(int)

# 8.3 Long-term contract flag
df["LongTermContract"] = (
    df.get("Contract_One year", 0) + df.get("Contract_Two year", 0)
).clip(upper=1)

# 8.4 Service count (dependency score)
service_cols = [
    "OnlineSecurity_Yes", "OnlineBackup_Yes",
    "DeviceProtection_Yes", "TechSupport_Yes",
    "StreamingTV_Yes", "StreamingMovies_Yes"
]

service_cols = [col for col in service_cols if col in df.columns]
df["ServiceCount"] = df[service_cols].sum(axis=1)

# ===============================
# 9. Final checks
# ===============================
print("Final dataset shape:", df.shape)
print("Missing values:\n", df.isnull().sum())

# ===============================
# 10. Save final dataset
# ===============================
df.to_csv("final_churn_data_v2.csv", index=False)
print("✅ Final dataset saved as final_churn_data_v2.csv")
