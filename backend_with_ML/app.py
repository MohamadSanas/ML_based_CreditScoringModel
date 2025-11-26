from flask import Flask,request,jsonify
from flask_cors import CORS
import numpy as np
from src.new_data_to_df import input_dataFrame
from joblib import load
import pandas as pd
import shap

 
app = Flask(__name__)
CORS(app)

loan_status_model = load('model/credit_eligibility_model.joblib')

@app.route('/')
def home():
    return jsonify ({"message": "Credit Score Prediction API is running!"}) 

@app.route('/predict',methods=['POST'])
def predict_loan_status():
    try:
        data = request.get_json()
        print("Received data for prediction:", data)
        
        
        
        input_df = input_dataFrame(
            dependent=data['Dependent'],
            Education=data['Education'],
            income=data['person_income'],
            Loan_term=data['Loan_term'],
            credit_scr=data['credit_score'],        # match parameter name
            Employment=data['Employment'],
            loan_amount=data['loan_amnt'],
            bankAssets=data['bankAssets'],          # match parameter name
            residentialAssets=data['residential_assets'],  
            commercialAssets=data['commercial_assets']    
        )

        
        print("Input DataFrame for prediction:\n", input_df)

        
        pred = loan_status_model.predict(input_df)[0]
        
        
        result = 'loan approved' if pred ==1 else 'loan rejected'
        
        
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
        
    
    
    
if __name__ =="__main__":
    app.run(debug=True)
        
        