import numpy as np
import pandas as pd
import shap
from joblib import load


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

loan_percent_income_dict = {
    'loan_percent_income_bucket_Low': 0,
    'loan_percent_income_bucket_Medium': 0,
    'loan_percent_income_bucket_High': 0,
    'loan_percent_income_bucket_Very High': 0
}

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
    
def find_loan_percent_income_bucket (percent,loan_percent_income_dict):
    if percent<= 0.2 :
        bucket= "loan_percent_income_bucket_Low"
    elif percent <=0.5:
        bucket = "loan_percent_income_bucket_Medium"
    elif percent <= 0.8:
        bucket = "loan_percent_income_bucket_High"
    elif percent <= 1:
        bucket = "loan_percent_income_bucket_Very High"
    
    for key in loan_percent_income_dict.keys():
        loan_percent_income_dict[key] = 0
        
    if bucket in loan_percent_income_dict:
        loan_percent_income_dict[bucket] = 1
    return loan_percent_income_dict
    
# Example applicant input
def input_dataFrame(age,gender,income,exp,credit_scr,prev_loan,education,home_ownrship,loan_intent,loan_amnt,crd_hist,int_rate):
    gen_code = 1 if gender == ' Male' else 0
    prev_loan_code =1 if prev_loan=="Yes" else 0
    loan_percent_income =loan_amnt/income
     
    applicant_data = {
        'person_age': age,
        'person_gender': gen_code,
        'person_income': income,       
        'person_emp_exp': exp,             
        'credit_score': credit_scr,              
        'previous_loan_defaults_on_file': prev_loan_code,

        # Education (One-hot)
        **find_education(education, education_dict),

        # Home ownership (One-hot)
        **find_home_ownership(home_ownrship, home_ownership_dict),

        # Loan intent (One-hot)
        **find_loan_intent(loan_intent, loan_intent_dict),

        # Log-transformed numeric features
        'loan_amnt_log': np.log1p(loan_amnt),  
        'cb_person_cred_hist_length_log': np.log1p(crd_hist), 
        'loan_int_rate_log': np.log1p(int_rate), 
        'loan_percent_income_log': np.log1p(loan_percent_income),  

        # Income bucket (One-hot)
        **find_loan_percent_income_bucket(loan_percent_income,loan_percent_income_dict)
    }
    # Convert to DataFrame
    input_df = pd.DataFrame([applicant_data])
    
    return input_df



