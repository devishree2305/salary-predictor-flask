from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
model = joblib.load('salary_predictor_with_field.pkl')

# Setup field encoding (very important: same as training)
fields_list = [
    'Data Science / AI / ML',
    'Software Engineering',
    'Marketing',
    'Finance',
    'Mechanical Engineering'
]
encoder = LabelEncoder()
encoder.fit(fields_list)

@app.route('/')
def home():
    return "Salary Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict_salary():
    data = request.get_json()

    # Read input
    years_experience = data['years_experience']
    field = data['field']

    # Encode field
    field_encoded = encoder.transform([field])[0]
    input_data = np.array([[years_experience, field_encoded]])

    # Predict
    prediction = model.predict(input_data)

    return jsonify({'predicted_salary': float(prediction[0])})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
