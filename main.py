from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model if it exists
model = None
if os.path.exists('churn_model.pkl'):
    try:
        with open('churn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    print("Model file not found. Using simple prediction logic.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        contract = request.form.get('contract', 'Month-to-month')
        tenure = int(request.form.get('tenure', 0))
        monthly_charges = float(request.form.get('monthly_charges', 0.0))
        total_charges = float(request.form.get('total_charges', 0.0))
        
        # Check if all values are zero
        if tenure == 0 and monthly_charges == 0.0 and total_charges == 0.0:
            return render_template('index.html', prediction_text="Please enter valid customer data (all fields cannot be zero)")
        
        if model is not None:
            # Use the trained model for prediction
            # Create features array based on the model's expected input
            # This is a simplified version - you may need to adjust based on your model
            features = np.array([[tenure, monthly_charges, total_charges]])
            prediction = model.predict(features)[0]
            prediction_text = "Customer will churn" if prediction == 1 else "Customer will not churn"
        else:
            # Simple prediction logic when model is not available
            churn_score = 0
            
            # Factors that increase churn risk
            if contract == 'Month-to-month':
                churn_score += 0.3
            if tenure < 12:
                churn_score += 0.2
            if monthly_charges > 80:
                churn_score += 0.2
            if total_charges < monthly_charges * 6:  # Low total charges relative to monthly
                churn_score += 0.1
                
            # Simple threshold-based prediction
            prediction = churn_score > 0.3
            
            if prediction:
                prediction_text = "High risk of churn"
            else:
                prediction_text = "Low risk of churn"
        
        return render_template('index.html', prediction_text=prediction_text)
        
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
