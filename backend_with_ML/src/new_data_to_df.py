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
def find_education(education_type):
    local_dict = {
        'person_education_Associate': 0,
        'person_education_Bachelor': 0,
        'person_education_Doctorate': 0,
        'person_education_High School': 0,
        'person_education_Master': 0
    }
    key = f"person_education_{education_type}"
    if key in local_dict:
        local_dict[key] = 1
    print("Education one-hot:", local_dict)
    return local_dict

def find_home_ownership(own_type):
    local_dict = {
        'person_home_ownership_MORTGAGE': 0,
        'person_home_ownership_OTHER': 0,
        'person_home_ownership_OWN': 0,
        'person_home_ownership_RENT': 0
    }
    key = f"person_home_ownership_{own_type}"
    if key in local_dict:
        local_dict[key] = 1
    print("Home ownership one-hot:", local_dict)
    return local_dict

def find_loan_intent(intent):
    local_dict = {
        'loan_intent_DEBTCONSOLIDATION': 0,
        'loan_intent_EDUCATION': 0,
        'loan_intent_HOMEIMPROVEMENT': 0,
        'loan_intent_MEDICAL': 0,
        'loan_intent_PERSONAL': 0,
        'loan_intent_VENTURE': 0
    }
    key = f"loan_intent_{intent}"
    if key in local_dict:
        local_dict[key] = 1
    print("Loan intent one-hot:", local_dict)
    return local_dict

def find_loan_percent_income_bucket(percent):
    local_dict = {
        'loan_percent_income_bucket_Low': 0,
        'loan_percent_income_bucket_Medium': 0,
        'loan_percent_income_bucket_High': 0,
        'loan_percent_income_bucket_Very High': 0
    }
    if percent <= 0.2:
        bucket = "loan_percent_income_bucket_Low"
    elif percent <= 0.5:
        bucket = "loan_percent_income_bucket_Medium"
    elif percent <= 0.8:
        bucket = "loan_percent_income_bucket_High"
    else:
        bucket = "loan_percent_income_bucket_Very High"
    
    local_dict[bucket] = 1
    print("Loan percent income bucket one-hot:", local_dict)
    return local_dict

def input_dataFrame(age, gender, income, exp, credit_scr, prev_loan,
                    education, home_ownership, loan_intent,
                    loan_amnt, crd_hist, int_rate):
    gen_code = 1 if gender.strip() == 'Male' else 0
    prev_loan_code = 1 if prev_loan.strip() == "Yes" else 0
    loan_percent_income = loan_amnt / income if income != 0 else 0

    applicant_data = {
        'person_age': age,
        'person_gender': gen_code,
        'person_income': income,
        'person_emp_exp': exp,
        'credit_score': credit_scr,
        'previous_loan_defaults_on_file': prev_loan_code,
        **find_education(education),
        **find_home_ownership(home_ownership),
        **find_loan_intent(loan_intent),
        'loan_amnt_log': np.log1p(loan_amnt),
        'cb_person_cred_hist_length_log': np.log1p(crd_hist),
        'loan_int_rate_log': np.log1p(int_rate),
        'loan_percent_income_log': np.log1p(loan_percent_income),
        **find_loan_percent_income_bucket(loan_percent_income)
    }

    input_df = pd.DataFrame([applicant_data])
    print("Converted DataFrame:\n", input_df)
    return input_df
