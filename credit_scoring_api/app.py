from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('C:/Users/User/Downloads/film/Credit-Scoring-Model-for-Consumer-Loans-in-Banking/data/Decision_tree.pkl')

@app.route('/')  # Home route
def home():
    return "Welcome to the Credit Scoring API!"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print("POST request received!")
        # Get the JSON data from the POST request
        data = request.get_json()
        # Convert the data into a pandas DataFrame (adjust the input to match your model)
        input_data = pd.DataFrame([data])
        # Make a prediction using the loaded model
        prediction = model.predict(input_data)
        # Return the prediction in a JSON format
        return jsonify({'prediction': int(prediction[0])})
    else:
        print("GET request received!")

if __name__ == '__main__':
    app.run(debug=True)