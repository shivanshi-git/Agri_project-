import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# --- INITIALIZATION ---
app = Flask(__name__)

# --- ROUTES ---

@app.route('/')
def home():
    """
    Renders the home page of the web application.
    """
    return render_template('index.html')  # Make sure templates/index.html exists


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print("Received data:", data)

        # Extract features
        N = data['N']
        P = data['P']
        K = data['K']
        temperature = data['temperature']
        humidity = data['humidity']
        ph = data['ph']
        rainfall = data['rainfall']

        # Prepare input for model
        input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Use a model (RandomForest recommended)
        prediction = RF.predict(input_features)
        predicted_class = prediction[0]

        # Decode predicted class index to crop name
        crop_name = crop_dict.get(predicted_class, "Unknown")

        return jsonify({"crop_recommendation": crop_name})

    except Exception as e:
        return jsonify({"error": str(e)})


# --- MODEL LOADING ---

try:
    with open('models/DecisionTree.pkl', 'rb') as file:
        DecisionTree = pickle.load(file)
    with open('models/NBClassifier.pkl', 'rb') as file:
        NaiveBayes = pickle.load(file)
    with open('models/SVMClassifier.pkl', 'rb') as file:
        SVM = pickle.load(file)
    with open('models/LogisticRegression.pkl', 'rb') as file:
        LogReg = pickle.load(file)
    with open('models/RandomForest.pkl', 'rb') as file:
        RF = pickle.load(file)
    print("All models loaded successfully!")
except FileNotFoundError:
    print("Error: Model files not found. Please ensure the 'models' directory and its contents are present.")
    exit()


try:
    df_labels = pd.read_csv('Crop_recommendation.csv')
    le = LabelEncoder()
    le.fit_transform(df_labels['label'])
    crop_dict = dict(zip(le.transform(le.classes_), le.classes_))
    print("Label Encoder recreated successfully!")
except FileNotFoundError:
    print("Error: 'Crop_recommendation.csv' not found. Cannot recreate Label Encoder.")
    exit()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
