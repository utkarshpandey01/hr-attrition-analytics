import os
import traceback
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.tree import DecisionTreeClassifier

# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)
CORS(app)

# ==========================================
# GLOBAL VARIABLES
# ==========================================

MODEL = None
FEATURE_COLUMNS = None
BASE_COLUMNS = None


# ==========================================
# INITIALIZE MODEL
# ==========================================

def init_model():
    global MODEL, FEATURE_COLUMNS, BASE_COLUMNS

    dataset_path = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Drop unnecessary columns
    df.drop(
        columns=[
            "EmployeeCount",
            "EmployeeNumber",
            "Over18",
            "StandardHours"
        ],
        inplace=True,
        errors='ignore'
    )

    # Convert target
    df["Attrition"] = df["Attrition"].map({
        "Yes": 1,
        "No": 0
    })

    # Features + target
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    # Save original columns
    BASE_COLUMNS = X.columns.tolist()

    # Encode training data
    X_encoded = pd.get_dummies(X)

    FEATURE_COLUMNS = X_encoded.columns.tolist()

    # Train model
    MODEL = DecisionTreeClassifier(
        max_depth=5,
        class_weight='balanced',
        random_state=42
    )

    MODEL.fit(X_encoded, y)

    print("✅ Model initialized successfully")


# Initialize on startup
init_model()


# ==========================================
# HEALTH CHECK
# ==========================================

@app.route('/')
def home():
    return jsonify({
        "status": "Backend Running Successfully"
    })


# ==========================================
# PREDICTION ROUTE
# ==========================================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        data = request.get_json()

        print("Incoming Data:")
        print(data)

        # ==========================================
        # USER INPUT
        # ==========================================

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

        # ==========================================
        # CREATE FULL DATAFRAME
        # ==========================================

        user_df = pd.DataFrame([user_row])

        # Add missing columns
        for col in BASE_COLUMNS:
            if col not in user_df.columns:
                user_df[col] = 0

        # Match exact training column order
        user_df = user_df[BASE_COLUMNS]

        # Encode
        user_encoded = pd.get_dummies(user_df)

        # Match training encoded columns
        user_encoded = user_encoded.reindex(
            columns=FEATURE_COLUMNS,
            fill_value=0
        )

        # ==========================================
        # PREDICTION
        # ==========================================

        prediction = int(MODEL.predict(user_encoded)[0])

        probability = MODEL.predict_proba(user_encoded)[0][1]

        risk_percentage = round(probability * 100, 1)

        print("Prediction Success")
        print("Risk:", risk_percentage)

        # ==========================================
        # RETURN RESPONSE
        # ==========================================

        return jsonify({
            "attrition_risk": prediction,
            "risk_percentage": risk_percentage
        })

    except Exception as e:

        print("======== ERROR ========")
        print(traceback.format_exc())

        return jsonify({
            "error": str(e)
        }), 500


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )