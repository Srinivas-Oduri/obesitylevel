import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import mode

# Load and preprocess the dataset
df = pd.read_csv("data/ObesityDataSet_raw_and_data_sinthetic.csv")
features = ['Age', 'Gender', 'Height', 'Weight', 'FCVC', 'NCP', 'FAF', 'CH2O', 'TUE', 'CALC']
label = 'NObeyesdad'

X = df[features]
y = df[label]

# Encode categorical
X = pd.get_dummies(X, columns=['Gender', 'CALC'])
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Train models
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    age = float(request.form['age'])
    gender = request.form['gender']
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    fcvc = float(request.form['fcvc'])
    ncp = float(request.form['ncp'])
    faf = float(request.form['faf'])
    ch2o = float(request.form['ch2o'])
    tue = float(request.form['tue'])
    calc = request.form['calc']

    # Convert to dataframe
    input_data = pd.DataFrame([[age, gender, height, weight, fcvc, ncp, faf, ch2o, tue, calc]],
                              columns=features)
    
    # One-hot encode
    input_data = pd.get_dummies(input_data)
    
    # Align with training data
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    
    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    dt_pred = dt.predict(input_scaled)
    rf_pred = rf.predict(input_scaled)
    ensemble_pred = mode([dt_pred, rf_pred])[0][0]

    result = le.inverse_transform([ensemble_pred])[0]
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
