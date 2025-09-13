
# from flask import Flask, request, jsonify
# import pickle
# import os
# import numpy as np
# import warnings

# warnings.filterwarnings('ignore')

# app = Flask(__name__)

# # Load the trained model and scaler from the 'models' directory
# try:
#     # We'll use the best model found in the notebook, XGBoost
#     # model_path = os.path.join(os.getcwd(), r'image\models', 'XGBoost.pkl')
#     #with open(model_path, 'rb') as file:
#     #    model = pickle.load(file)
#     #print("Model loaded successfully.")

#     # We also need the scaler used for the SVM model, though XGBoost doesn't strictly need it,
#     # it's good practice to have the same data pre-processing in the API as in training.
#     # The original notebook didn't save the scaler, so we'll re-fit it for demonstration.
#     # In a real-world scenario, you would save and load the scaler as well.
#     from sklearn.preprocessing import MinMaxScaler
#     df = pd.read_csv(r'image\Crop_recommendation.csv') # Re-load the data to fit the scaler
#     features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
#     scaler = MinMaxScaler().fit(features)
#     print("Scaler re-fitted successfully.")

# except (FileNotFoundError, IOError, pickle.PickleError) as e:
#     print(f"Error loading model: {e}")
#     model = None
#     scaler = None

# @app.route("/predict", methods=["POST"])
# def predict():
#     if model is None or scaler is None:
#         return jsonify({"error": "Model or scaler not loaded. Please check server logs."}), 500

#     try:
#         data = request.get_json(force=True)
#         required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
#         # Check for missing fields
#         for field in required_fields:
#             if field not in data:
#                 return jsonify({"error": f"Missing field: '{field}'"}), 400

#         # Create a numpy array from the JSON data
#         features = np.array([[
#             data['N'], data['P'], data['K'], data['temperature'],
#             data['humidity'], data['ph'], data['rainfall']
#         ]])

#         # The XGBoost model was trained on unscaled data, so we don't need to scale the input.
#         # But if you chose to use the SVM model, you would uncomment the next line:
#         # features_scaled = scaler.transform(features)

#         # Predict using the loaded model
#         prediction = model.predict(features)[0]

#         return jsonify({"recommended_crop": str(prediction)})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0', port=5000)
