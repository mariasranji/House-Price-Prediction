from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- ADD THIS LINE
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)  # <-- ADD THIS LINE TO ALLOW WEB UI CONNECTIONS
# 1. Load the model bundle at application startup
try:
    model_bundle = joblib.load('gradient_boosting_house_price_model.pkl')
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    saved_num_cols = model_bundle['important_num_cols']
    saved_cat_cols = model_bundle['cat_cols']
    model_columns = model_bundle['model_columns']
    print("Model bundle loaded successfully!")
except Exception as e:
    print(f"Error loading model bundle: {str(e)}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server side."}), 500

    # 2. Get JSON data from the request
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided. Request body must be JSON."}), 400

    try:
        # Convert incoming JSON data (either a single dict or a list of dicts) into a DataFrame
        if isinstance(data, dict):
            input_df = pd.DataFrame([data])
        elif isinstance(data, list):
            input_df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Invalid JSON format. Expected an object or an array of objects."}), 400

        # 3. Check for missing primary columns
        required_raw_cols = saved_num_cols + saved_cat_cols
        missing_cols = [col for col in required_raw_cols if col not in input_df.columns]
        if missing_cols:
            return jsonify({"error": f"Missing required columns in input data: {missing_cols}"}), 400

        # 4. Filter down to only the important columns used during feature selection
        processed_df = input_df[required_raw_cols].copy()

        # 5. Handle Categorical Encoding (One-Hot Encoding)
        processed_df = pd.get_dummies(processed_df, columns=saved_cat_cols)

        # 6. CRITICAL PIPELINE ALIGNMENT
        # Force the feature matrix to have the exact same columns (and order) as X during model.fit().
        # This fixes missing categories (fills with 0) or completely drops unseen categories.
        processed_df = processed_df.reindex(columns=model_columns, fill_value=0)

        # 7. Apply the saved StandardScaler scaling parameters
        processed_df[saved_num_cols] = scaler.transform(processed_df[saved_num_cols])

        # 8. Generate Predictions
        predictions = model.predict(processed_df)

        # 9. Return the results
        return jsonify({
            "status": "success",
            "predictions": predictions.tolist()
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred during preprocessing or prediction: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the application locally on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
