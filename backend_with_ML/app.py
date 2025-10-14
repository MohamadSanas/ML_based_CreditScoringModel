from flask import Flask,request,jsonify
from flask_cors import CORS
from src.new_data_to_df import input_dataFrame
import numpy as np
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
        
        input_df = input_dataFrame(
        age=data['person_age'],
        gender=data['person_gender'],
        income=data['person_income'],
        exp=data['person_emp_exp'],
        credit_scr=data['credit_score'],
        prev_loan=data['previous_loan_defaults_on_file'],
        education=data['education'],
        home_ownrship=data['home_ownership'],
        loan_intent=data['loan_intent'],
        loan_amnt=data['loan_amnt'],
        crd_hist=data['credit_history_length'],
        int_rate=data['loan_interest_rate']
        )
        
        pred = loan_status_model.predict(input_df)[0]
        
        
        result = 'loan approved' if pred ==1 else 'loan rejected'
        
        
        # SHAP explainability (optional)
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
        return e
    
    
    
if __name__ =="__main__":
    app.run(debug=True)
        
        