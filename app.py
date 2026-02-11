from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model & features
model = joblib.load("churn_model.joblib")
model_features = joblib.load("model_features.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = {}

    for feature in model_features:
        value = request.form.get(feature)
        input_data[feature] = float(value) if value else 0

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    result = "Customer Will Churn" if prediction == 1 else "Customer Will Stay"
    result_class = "churn" if prediction == 1 else "no-churn"

    return render_template(
        "index.html",
        prediction_text=result,
        probability_text=f"Churn Probability: {probability:.2%}",
        result_class=result_class
    )

if __name__ == "__main__":
    app.run(debug=True)
