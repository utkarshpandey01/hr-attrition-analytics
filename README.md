# 🚀 Enterprise HR Attrition Predictor

A futuristic AI-powered Employee Attrition Prediction Dashboard built using **Machine Learning**, **Flask**, and a premium **Glassmorphism UI**.

# 🌐 Live Demo : https://hr-attrition-analytics-predictor.netlify.app/

This project predicts whether an employee is likely to leave the company based on critical workplace and behavioral factors such as:

- Age
- Monthly Income
- Overtime
- Distance From Home
- Business Travel
- Years at Company
- Marital Status
- Total Working Years

The system uses a **Decision Tree Classifier** trained on real HR analytics data and delivers real-time risk predictions through an interactive dashboard.

---



## Backend API
https://hr-attrition-analytics.onrender.com

---

# ✨ Features

## 🔥 AI-Powered Attrition Prediction
Uses a Machine Learning Decision Tree model to estimate employee turnover probability.

---

## 📊 Interactive Analytics Dashboard
Modern glassmorphism-based futuristic UI with:

- Animated risk meter
- Dynamic sliders
- Live data processing
- Smart risk interventions
- Corporate analytics feel

---

## ⚡ Real-Time Prediction Engine
The frontend sends employee data to a Flask backend API that instantly returns:

- Attrition Risk
- Risk Percentage
- Retention Status

---

## 🧠 Decision Tree Machine Learning
Model trained using:

- Scikit-Learn
- Pandas
- HR Analytics Dataset

---

## 🎯 Smart Risk Detection
Highlights risky conditions such as:

- Overtime workload
- Low salary
- Long travel distance

---

# 🛠️ Tech Stack

## Frontend
- HTML5
- CSS3
- JavaScript
- Chart.js

---

## Backend
- Python
- Flask
- Flask-CORS

---

## Machine Learning
- Scikit-Learn
- Decision Tree Classifier
- Pandas
- NumPy

---

## Deployment
- Netlify (Frontend)
- Render (Backend)

---

# 🧠 Machine Learning Workflow

```txt
Dataset
   ↓
Data Cleaning
   ↓
Feature Encoding
   ↓
Decision Tree Training
   ↓
Flask API Deployment
   ↓
Frontend Integration
   ↓
Real-Time Prediction
```

---

# 📂 Project Structure

```bash
HR-ATTRITION-PROJECT
│
├── app.py
├── requirements.txt
├── runtime.txt
├── index.html
├── WA_Fn-UseC_-HR-Employee-Attrition.csv
└── README.md
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/hr-attrition-analytics.git
```

---

## 2️⃣ Move Into Project Folder

```bash
cd hr-attrition-analytics
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run Flask Server

```bash
python app.py
```

---

# 🌐 Backend API

## POST `/predict`

Predicts employee attrition probability.

---

## Request Body

```json
{
  "Age": 35,
  "MonthlyIncome": 75000,
  "YearsAtCompany": 5,
  "TotalWorkingYears": 10,
  "DistanceFromHome": 5,
  "OverTime": "No",
  "BusinessTravel": "Travel_Rarely",
  "Department": "Research & Development",
  "MaritalStatus": "Single"
}
```

---

## Response

```json
{
  "attrition_risk": 0,
  "risk_percentage": 18.7
}
```

---

# 📈 Dataset

Dataset used:

## IBM HR Analytics Employee Attrition Dataset

Contains employee-related features such as:

- Salary
- Education
- Department
- Overtime
- Job Role
- Work Experience
- Travel Frequency

Used for predicting employee retention patterns.

---

# 🚀 Deployment

## Frontend Deployment
Deployed on:

- Netlify

---

## Backend Deployment
Deployed on:

- Render

---

# 🔮 Future Improvements

- Deep Learning integration
- Employee recommendation engine
- PDF analytics report generation
- Authentication system
- Database integration
- Advanced HR dashboards
- Real-time charts
- Dark/Light mode toggle

---

# 🧪 Model Information

| Model | Accuracy |
|------|------|
| Decision Tree Classifier | ~80% |

---

# 👨‍💻 Author

## Utkarsh Pandey

Passionate about:

- Artificial Intelligence
- Machine Learning
- Full Stack Development
- Data Analytics

---

# ⭐ Support

If you liked this project:

- Star this repository ⭐
- Fork the project 🍴
- Share with others 🚀

---

# 📜 License

This project is licensed under the MIT License.

---

# 💡 Final Note

This project demonstrates how Machine Learning can be integrated with modern web technologies to create intelligent, real-world analytics systems capable of assisting HR departments in employee retention analysis.
