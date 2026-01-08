from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

# Load the pre-trained linear regression model
def load_linear_model():
    try:
        with open('student_mark_prdictor_model1.pkl', 'rb') as file:
            model = pickle.load(file)
        print("Linear model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading linear model: {e}")
        # Create a fallback model if file not found
        print("Creating fallback linear model...")
        return create_fallback_model()

def create_fallback_model():
    """Create a realistic linear model as fallback"""
    model = LinearRegression()
    # Realistic relationship: 4 marks per study hour, starting from 0, max 100
    X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [15], [20], [24]])
    y = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 60, 80, 96])
    model.fit(X, y)
    return model

# Initialize polynomial regression model
def create_polynomial_model(degree=2):
    return PolynomialFeatures(degree=degree)

# Load sample data for demonstration
def load_sample_data():
    try:
        return pd.read_csv('low_study_predictions.csv')
    except:
        # Create realistic sample data
        return pd.DataFrame({
            'Study_Hours': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 24],
            'Predicted_Marks': [0, 10, 20, 30, 35, 40, 50, 55, 60, 65, 70, 80, 90, 95],
            'Status': ['FAIL', 'FAIL', 'FAIL', 'FAIL', 'FAIL', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS']
        })

linear_model = load_linear_model()
sample_data = load_sample_data()

def calculate_marks(study_hours, algorithm):
    """Calculate marks with realistic constraints"""
    # Special case: 0 study hours should result in 0 marks and FAIL
    if study_hours == 0:
        return 0, "FAIL"
    
    if algorithm == 'polynomial':
        # Create polynomial features
        poly = create_polynomial_model(degree=2)
        
        # Use sample data to fit polynomial model
        X_sample = sample_data[['Study_Hours']].values
        y_sample = sample_data['Predicted_Marks'].values
        
        X_poly = poly.fit_transform(X_sample)
        poly_model = LinearRegression()
        poly_model.fit(X_poly, y_sample)
        
        # Make prediction
        study_hours_poly = poly.transform([[study_hours]])
        prediction = poly_model.predict(study_hours_poly)
        predicted_mark = max(0, min(100, round(prediction[0], 2)))  # Constrain between 0-100
    else:
        # Use linear regression
        prediction = linear_model.predict([[study_hours]])
        predicted_mark = max(0, min(100, round(prediction[0], 2)))  # Constrain between 0-100
    
    # Determine status (passing threshold: 40 marks)
    status = "PASS" if predicted_mark >= 33 else "FAIL"
    
    return predicted_mark, status

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        study_hours = float(request.form['study_hours'])
        algorithm = request.form.get('algorithm', 'linear')
        
        # Validate study hours
        if study_hours < 0 or study_hours > 24:
            return render_template('index.html', 
                                 prediction_text='Error: Study hours must be between 0 and 24',
                                 study_hours=study_hours,
                                 algorithm=algorithm)
        
        predicted_mark, status = calculate_marks(study_hours, algorithm)
        
        return render_template('index.html', 
                             prediction_text=f'Predicted Mark: {predicted_mark}%',
                             status=status,
                             study_hours=study_hours,
                             algorithm=algorithm)
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}',
                             study_hours=request.form.get('study_hours', ''))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        study_hours = float(data['study_hours'])
        algorithm = data.get('algorithm', 'linear')
        
        # Validate study hours
        if study_hours < 0 or study_hours > 24:
            return jsonify({'error': 'Study hours must be between 0 and 24'}), 400
        
        predicted_mark, status = calculate_marks(study_hours, algorithm)
        
        return jsonify({
            'study_hours': study_hours,
            'predicted_mark': predicted_mark,
            'status': status,
            'algorithm': algorithm
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)