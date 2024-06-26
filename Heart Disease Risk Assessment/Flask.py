# Importing necessary libraries
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

# Define route for the home page
@app.route('/')
def home():
    # Render the home page template with initial values and placeholders
    return render_template('index.html', sysBP='', BMI='', totChol='', age='', glucose='', prediction_text='', error_message='')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extracting values from the form
            sysBP = float(request.form['sysBP'])
            BMI = float(request.form['BMI'])
            totChol = float(request.form['totChol'])
            age = float(request.form['age'])
            glucose = float(request.form['glucose'])

            # Creating a numpy array with the input values
            input_features = np.array([[sysBP, BMI, totChol, age, glucose]])
            input_features = pd.DataFrame(input_features)

            # Scaling the input features
            input_features_scaled = scaler.transform(input_features)

            # Making prediction using the model
            prediction = model.predict(input_features_scaled)

            # Interpret prediction and set prediction text
            if prediction[0] == 1:
                prediction_text = f'You are at risk of CHD'
            else:
                prediction_text = f'You are not at risk of CHD'

            # Render the home page template with prediction result and input values
            return render_template('index.html', prediction_text=prediction_text, sysBP=sysBP, BMI=BMI, totChol=totChol, age=age, glucose=glucose)

        except Exception as e:
            # Render the home page template with error message if an error occurs
            error_message = f'An error occurred while predicting: {str(e)}'
            return render_template('index.html', error_message=error_message)

    else:
        # Render the home page template with error message if request method is invalid
        error_message = 'Invalid request method.'
        return render_template('index.html', error_message=error_message)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=8080)
