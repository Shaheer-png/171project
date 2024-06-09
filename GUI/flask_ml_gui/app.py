from flask import Flask, render_template, request
from model import predict
from feature_extraction import get_feature_names
import joblib

app = Flask(__name__)

# Load feature names for the form
features = joblib.load('feature_names.pkl')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def prediction():
    # Ensure all features are present in the input data
    user_input = {feature: request.form.get(feature, 0) for feature in features}
    
    # Debugging: Print the user input to check if all features are captured correctly
    print("User Input:", user_input)
    
    prediction = predict(user_input)  # Use the actual predict function
    
    # Debugging: Print the raw prediction result
    print("Raw Prediction:", prediction)
    
    return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
