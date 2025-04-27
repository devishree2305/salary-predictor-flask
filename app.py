from flask import Flask, render_template, request
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model and the encoder
model = joblib.load('salary_predictor_with_field.pkl')
encoder = joblib.load('field_encoder.pkl')

# Route for the homepage (Input Form)
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input from the form
    years_experience = float(request.form['years_experience'])
    field = request.form['field']
    
    # Encode the field to match the model's training
    field_encoded = encoder.transform([field])[0]
    
    # Prepare input data for prediction
    input_data = np.array([[years_experience, field_encoded]])
    
    # Make prediction
    predicted_salary = model.predict(input_data)[0]
    
    # Return result to the frontend
    return render_template('index.html', prediction_text=f"Predicted Salary: ₹{predicted_salary:,.2f}")

if __name__ == "__main__":
    app.run(debug=True)
