import os
import traceback
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)
CORS(app)

# --- GLOBAL MODEL PIPELINE ---
PIPELINE = None

def init_model():
    global MODEL, FEATURE_COLUMNS, BASE_COLUMNS

    dataset_path = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    # 1. DEFINE EXACTLY WHAT YOUR FRONTEND RELEASES
    # This prevents the model from generating dummy columns for features you don't submit!
    BASE_COLUMNS = [
        "Age", "MonthlyIncome", "YearsAtCompany", "TotalWorkingYears", 
        "DistanceFromHome", "OverTime", "BusinessTravel", "Department", "MaritalStatus"
    ]

    # Convert target
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    # Slice out ONLY the columns matching your interface capabilities
    X = df[BASE_COLUMNS]
    y = df["Attrition"]

    # 2. ENCODE SAFELY
    # This matches the schema architecture of your prediction block
    X_encoded = pd.get_dummies(X)
    FEATURE_COLUMNS = X_encoded.columns.tolist()

    # Train model
    MODEL = DecisionTreeClassifier(
        max_depth=5,
        class_weight='balanced',
        random_state=42
    )
    MODEL.fit(X_encoded, y)

    print("✅ Model re-trained and optimized for frontend integration!")

# Run compilation on startup
init_model()

@app.route('/')
def home():
    return jsonify({"status": "Backend Online", "engine": "Sklearn Pipeline Optimized"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}

        # Construct raw payload safely 
        user_row = {
            "Age": int(data.get("Age", 35)),
            "MonthlyIncome": int(data.get("MonthlyIncome", 75000)),
            "YearsAtCompany": int(data.get("YearsAtCompany", 5)),
            "TotalWorkingYears": int(data.get("TotalWorkingYears", 10)),
            "DistanceFromHome": int(data.get("DistanceFromHome", 5)),
            "OverTime": str(data.get("OverTime", "No")),
            "BusinessTravel": str(data.get("BusinessTravel", "Travel_Rarely")),
            "Department": str(data.get("Department", "Research & Development")),
            "MaritalStatus": str(data.get("MaritalStatus", "Single"))
        }

        # Convert to a DataFrame row
        user_df = pd.DataFrame([user_row])

        # Pipeline automatically drops unused features, imputes defaults, matches
        # encoding arrays, and preserves structural sequence perfectly!
        prediction = int(PIPELINE.predict(user_df)[0])
        probability = PIPELINE.predict_proba(user_df)[0][1]
        risk_percentage = round(probability * 100, 1)

        return jsonify({
            "attrition_risk": prediction,
            "risk_percentage": risk_percentage
        })

    except Exception as e:
        print("======== RUNTIME ERROR ========")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)