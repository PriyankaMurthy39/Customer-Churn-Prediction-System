import pickle
from model_logistic import model  # adjust if model variable name is different

# Save model using compatible protocol
with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ Model re-saved successfully")
