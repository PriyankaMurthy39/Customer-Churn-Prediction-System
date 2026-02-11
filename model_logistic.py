import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Load data
df = pd.read_csv("final_churn_data_v2.csv")

print(df.head())
print(df.shape)

# Target
y = df["Churn"]

# Features
X = df.drop("Churn", axis=1)

print(X.shape)
print(y.value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(X_test.shape)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model training completed")

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc_auc)

# Save predictions for Power BI
results = X_test.copy()
results["ActualChurn"] = y_test.values
results["PredictedChurn"] = y_pred
results["ChurnProbability"] = y_prob

results.to_csv("churn_predictions.csv", index=False)
print("Predictions saved to churn_predictions.csv")

# Feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

feature_importance["AbsCoefficient"] = feature_importance["Coefficient"].abs()
feature_importance = feature_importance.sort_values(
    by="AbsCoefficient",
    ascending=False
)

feature_importance.to_csv("feature_importance_logistic.csv", index=False)
print("Feature importance saved")

print(feature_importance.head(10))

# ✅ SAVE MODEL & FEATURES (ONLY ONCE, AT THE END)
joblib.dump(model, "churn_model.joblib")
joblib.dump(X.columns.tolist(), "model_features.joblib")

print("✅ Model and features saved using joblib")
