import os
import traceback
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# ==========================================
# FLASK INIT
# ==========================================

app = Flask(__name__)
CORS(app)

PIPELINE = None

# ==========================================
# TRAIN MODEL
# ==========================================

def init_model():

    global PIPELINE

    dataset_path = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset not found")

    df = pd.read_csv(dataset_path)

    # ==========================================
    # TARGET
    # ==========================================

    df["Attrition"] = df["Attrition"].map({
        "Yes": 1,
        "No": 0
    })

    # ==========================================
    # FEATURES
    # ==========================================

    FEATURES = [
        "Age",
        "MonthlyIncome",
        "YearsAtCompany",
        "TotalWorkingYears",
        "DistanceFromHome",
        "OverTime",
        "BusinessTravel",
        "Department",
        "MaritalStatus"
    ]

    X = df[FEATURES]
    y = df["Attrition"]

    # ==========================================
    # CATEGORICAL + NUMERICAL
    # ==========================================

    categorical_features = [
        "OverTime",
        "BusinessTravel",
        "Department",
        "MaritalStatus"
    ]

    numerical_features = [
        "Age",
        "MonthlyIncome",
        "YearsAtCompany",
        "TotalWorkingYears",
        "DistanceFromHome"
    ]

    # ==========================================
    # PREPROCESSOR
    # ==========================================

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown='ignore'),
                categorical_features
            ),
            (
                "num",
                "passthrough",
                numerical_features
            )
        ]
    )

    # ==========================================
    # MODEL
    # ==========================================

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )

    # ==========================================
    # FULL PIPELINE
    # ==========================================

    PIPELINE = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # ==========================================
    # TRAIN TEST SPLIT
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ==========================================
    # TRAIN
    # ==========================================

    PIPELINE.fit(X_train, y_train)

    # ==========================================
    # ACCURACY
    # ==========================================

    preds = PIPELINE.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print(f"✅ Model Accuracy: {acc * 100:.2f}%")

# ==========================================
# INITIALIZE
# ==========================================

init_model()

# ==========================================
# HOME ROUTE
# ==========================================

@app.route('/')
def home():
    return jsonify({
        "status": "Backend Online",
        "model": "Random Forest Pipeline"
    })

# ==========================================
# PREDICT ROUTE
# ==========================================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        data = request.get_json()

        user_data = pd.DataFrame([{
            "Age": int(data["Age"]),
            "MonthlyIncome": int(data["MonthlyIncome"]),
            "YearsAtCompany": int(data["YearsAtCompany"]),
            "TotalWorkingYears": int(data["TotalWorkingYears"]),
            "DistanceFromHome": int(data["DistanceFromHome"]),
            "OverTime": data["OverTime"],
            "BusinessTravel": data["BusinessTravel"],
            "Department": data["Department"],
            "MaritalStatus": data["MaritalStatus"]
        }])

        # ==========================================
        # PREDICTION
        # ==========================================

        prediction = int(
            PIPELINE.predict(user_data)[0]
        )

        probability = float(
            PIPELINE.predict_proba(user_data)[0][1]
        )

        risk_percentage = round(probability * 100, 1)

        return jsonify({
            "attrition_risk": prediction,
            "risk_percentage": risk_percentage
        })

    except Exception as e:

        print(traceback.format_exc())

        return jsonify({
            "error": str(e)
        }), 500

# ==========================================
# RUN SERVER
# ==========================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )