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
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Securely capture JSON data from Netlify fetch call
        data = request.json or {}
        
        # 1. Capture user features with safe type handling
        user_row = {
            'Age': int(data.get('Age', 35)),
            'MonthlyIncome': int(data.get('MonthlyIncome', 5000)),
            'YearsAtCompany': int(data.get('YearsAtCompany', 5)),
            'TotalWorkingYears': int(data.get('TotalWorkingYears', 10)),
            'DistanceFromHome': int(data.get('DistanceFromHome', 5)),
            'OverTime': str(data.get('OverTime', 'No')),
            'BusinessTravel': str(data.get('BusinessTravel', 'Travel_Rarely')),
            'Department': str(data.get('Department', 'Research & Development')),
            'MaritalStatus': str(data.get('MaritalStatus', 'Single'))
        }
        
        # 2. Convert user row to data frame matching raw dataset format
        user_df = pd.DataFrame([user_row])
        base_df = X_RAW_TEMPLATE.copy()
        
        # 3. Use robust pandas alignment to append rows without breaking types
        temp_df = pd.concat([base_df, user_df], ignore_index=True)
        
        # Safe filling strategy: fill missing categories with mode, numeric with median
        for col in temp_df.columns:
            if temp_df[col].isnull().any():
                if temp_df[col].dtype == 'object':
                    temp_df[col] = temp_df[col].fillna(base_df[col].mode()[0])
                else:
                    temp_df[col] = temp_df[col].fillna(base_df[col].median())
        
        # 4. Generate identical structural dummies 
        temp_encoded = pd.get_dummies(temp_df, drop_first=True)
        user_encoded = temp_encoded.iloc[[-1]].reindex(columns=FEATURE_COLUMNS, fill_value=0)
        
        # 5. Run Decision Tree inference 
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
        import traceback
        print("=== DEPLOYMENT CRASH TRACE ===")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)