import numpy as np
import pandas as pd
import shap
from joblib import load

# Load model
loan_status_model = load("model/credit_eligibility_model.joblib")
loan_amnt_model = load("model/loan_amount_model.joblib")

education_dict = {'person_education_Associate': 0,
    'person_education_Bachelor': 0,
    'person_education_Doctorate': 0,
    'person_education_High School': 0,
    'person_education_Master': 0}

home_ownership_dict = {'person_home_ownership_MORTGAGE': 0,
    'person_home_ownership_OTHER': 0,
    'person_home_ownership_OWN': 0,
    'person_home_ownership_RENT': 0}

loan_intent_dict= {'loan_intent_DEBTCONSOLIDATION': 0,
    'loan_intent_EDUCATION': 0,
    'loan_intent_HOMEIMPROVEMENT': 0,
    'loan_intent_MEDICAL': 0,
    'loan_intent_PERSONAL': 0,
    'loan_intent_VENTURE': 0}

def find_education(education_type,education_dict):
    education_type=f"person_education_{education_type}"
    for key in education_dict.keys():
        education_dict[key] = 0
    
    if education_type in education_dict:
        education_dict[education_type]=1    
    return education_dict

def find_home_ownership(own_type,own_type_dict):
    own_type=f"person_home_ownership_{own_type}"
    for key in own_type_dict.keys():
        own_type_dict[key]=0
    if own_type in own_type_dict:
        own_type_dict[own_type]=1
    return own_type_dict

def find_loan_intent(intent,loan_intent_dict):
    intent=f"loan_intent_{intent}"
    for key in loan_intent_dict.keys():
        loan_intent_dict[key]=0
    if intent in loan_intent_dict:
        loan_intent_dict[intent]=1
        
    return loan_intent_dict
    
    
# Example applicant input
applicant_data = {
    'person_age': 35,                 # older age
    'person_gender': 0,
    'person_income': 1000000,         # higher income
    'person_emp_exp': 10,             # longer employment
    'credit_score': 750,              # high credit score
    'previous_loan_defaults_on_file': 0,

    # Education (One-hot)
    **find_education("Master", education_dict),

    # Home ownership (One-hot)
    **find_home_ownership("MORTGAGE", home_ownership_dict),

    # Loan intent (One-hot)
    **find_loan_intent("PERSONAL", loan_intent_dict),

    # Log-transformed numeric features
    'loan_amnt_log': np.log1p(10000),  # smaller loan relative to income
    'cb_person_cred_hist_length_log': np.log1p(10),  # longer credit history
    'loan_int_rate_log': np.log1p(20),  # moderate interest rate
    'loan_percent_income_log': np.log1p(0.01),  # low % of income

    # Income bucket (One-hot)
    'loan_percent_income_bucket_Low': 0,
    'loan_percent_income_bucket_Medium': 1,
    'loan_percent_income_bucket_High': 0,
    'loan_percent_income_bucket_Very High': 0
}



# Convert to DataFrame
input_df = pd.DataFrame([applicant_data])

"""
for col in input_df.columns:
    print(f"{col}: {input_df[col].iloc[0]}")
"""

input_df_loan_amnt = input_df.drop(columns=['previous_loan_defaults_on_file','loan_amnt_log'], errors='ignore')
input_df__loan_amnt = input_df_loan_amnt[loan_amnt_model.feature_names_in_]


# Make prediction
pred = loan_status_model.predict(input_df)[0]
print("Predicted loan eligibility:", "Approved" if pred == 1 else "Rejected")



explainer = shap.TreeExplainer(loan_status_model)
shap_values = explainer.shap_values(input_df)

# Get SHAP values as DataFrame
shap_df = pd.DataFrame({
    'Feature': input_df.columns,
    'SHAP_value': shap_values[0]
}).sort_values(by='SHAP_value', ascending=False)

shap_dict = shap_df.set_index('Feature')['SHAP_value'].to_dict()

shape_approval_loan= {}
shape_rejected_loan = {}

for feature,values in shap_dict.items():
    if values>=0.01:
        shape_approval_loan[feature] = values
    elif values< -0.01:
        shape_rejected_loan[feature]=values



if pred == 1:
    for feature in shape_approval_loan:  # top positive SHAP features
        if feature in applicant_data:    # check if this feature exists in input
            print(feature)


if pred ==0 :
    for feature in shape_rejected_loan:
        if feature in applicant_data:
            print (feature)


