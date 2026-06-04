import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.tree import DecisionTreeClassifier

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# --- GLOBAL MODEL VARIABLES ---
MODEL = None
FEATURE_COLUMNS = []
X_RAW_TEMPLATE = None

def init_model():
    global MODEL, FEATURE_COLUMNS, X_RAW_TEMPLATE
    
    # Load dataset 
    dataset_path = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Could not find the dataset file: {dataset_path}")
        
    df = pd.read_csv(dataset_path)
    
    # Drop uninformative columns
    df = df.drop(["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"], axis=1, errors='ignore')
    
    if df['Attrition'].dtype == 'object':
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        
    X_raw = df.drop("Attrition", axis=1)
    y = df["Attrition"]
    
    # Train Model on encoded data
    X_encoded = pd.get_dummies(X_raw, drop_first=True)
    
    MODEL = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
    MODEL.fit(X_encoded, y)
    
    FEATURE_COLUMNS = X_encoded.columns.tolist()
    X_RAW_TEMPLATE = X_raw 

# Run the model setup on server boot
init_model()

# --- ADDED THIS TO FIX THE 404 NOT FOUND ERROR WHEN CLICKING THE LINK ---
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "online",
        "message": "Backend server is running perfectly! Do not open the /predict link in your browser directly. Use your frontend UI to send data here."
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        user_row = {
            'Age': int(data.get('Age', 35)),
            'MonthlyIncome': int(data.get('MonthlyIncome', 5000)),
            'YearsAtCompany': int(data.get('YearsAtCompany', 5)),
            'TotalWorkingYears': int(data.get('TotalWorkingYears', 10)),
            'DistanceFromHome': int(data.get('DistanceFromHome', 5)),
            'OverTime': data.get('OverTime', 'No'),
            'BusinessTravel': data.get('BusinessTravel', 'Travel_Rarely'),
            'Department': data.get('Department', 'Research & Development'),
            'MaritalStatus': data.get('MaritalStatus', 'Single')
        }
        
        base_df = X_RAW_TEMPLATE.copy()
        
        for col in base_df.columns:
            if col not in user_row:
                if base_df[col].dtype == 'object':
                    user_row[col] = base_df[col].mode()[0]
                else:
                    user_row[col] = int(base_df[col].mean())
                    
        temp_df = pd.concat([base_df, pd.DataFrame([user_row])], ignore_index=True)
        temp_encoded = pd.get_dummies(temp_df, drop_first=True)
        user_encoded = temp_encoded.iloc[[-1]].reindex(columns=FEATURE_COLUMNS, fill_value=0)
        
        prediction = int(MODEL.predict(user_encoded)[0])
        probabilities = MODEL.predict_proba(user_encoded)[0]
        risk_percentage = round(probabilities[1] * 100, 1)
        
        importances = dict(zip(FEATURE_COLUMNS, MODEL.feature_importances_))
        
        return jsonify({
            'attrition_risk': prediction,
            'risk_percentage': risk_percentage,
            'feature_weights': importances 
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)