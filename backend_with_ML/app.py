import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import shap
from joblib import load
from src.new_data_to_df import input_dataFrame

app = Flask(__name__, static_folder='frontend')
CORS(app)

# ===========================
# Serve Flutter Web Build
# ===========================

# Serve all static files (JS, CSS, images, assets…)
@app.route('/<path:path>', methods=['GET'])
def static_proxy(path):
    return send_from_directory('frontend', path)

# Serve the Flutter index.html
@app.route('/', methods=['GET'])
def root():
    return send_from_directory('frontend', 'index.html')

# ===========================
# Backend API Routes
# ===========================

loan_status_model = load('model/credit_eligibility_model.joblib')

@app.route('/predict', methods=['POST'])
def predict_loan_status():
    try:
        data = request.get_json()
        print("Received data for prediction:", data)

        input_df = input_dataFrame(
            dependent=data['Dependent'],
            Education=data['Education'],
            income=data['person_income'],
            Loan_term=data['Loan_term'],
            credit_scr=data['credit_score'],
            Employment=data['Employment'],
            loan_amount=data['loan_amnt'],
            bankAssets=data['bankAssets'],
            residentialAssets=data['residential_assets'],
            commercialAssets=data['commercial_assets']
        )

        print("Input DataFrame for prediction:\n", input_df)

        pred = loan_status_model.predict(input_df)[0]
        result = 'loan approved' if pred == 1 else 'loan rejected'

        # SHAP explainability
        explainer = shap.TreeExplainer(loan_status_model)
        shap_values = explainer.shap_values(input_df)
        shap_df = pd.DataFrame({
            'Feature': input_df.columns,
            'SHAP_value': shap_values[0]
        }).sort_values(by='SHAP_value', ascending=False)

        top_features = shap_df.head(5).to_dict(orient='records')

        return jsonify({
            "loan_status": result,
            "top_features": top_features
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check for Render
@app.route("/healthz")
def health():
    return "OK", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
