import numpy as np
import pandas as pd
import shap
from joblib import load


def input_dataFrame(dependent, Education, income, Loan_term, credit_scr, Employment,
                    loan_amount, bankAssets, residentialAssets, commercialAssets):
    # Encode categorical features
    Education_code = 1 if Education.strip().lower() == 'yes' else 0
    Employment_code = 1 if Employment.strip().lower() == "employed" else 0

    # Build dictionary matching the model's trained feature names
    applicant_data = {
        'no_of_dependents': dependent,
        'education': Education_code,
        'self_employed': Employment_code,
        'income_annum': income,
        'loan_amount': loan_amount,
        'loan_term': Loan_term,
        'credit_score': credit_scr,
        'residential_assets_value': residentialAssets,  # match training column
        'commercial_assets_value': commercialAssets,    # match training column
        'bank_asset_value': bankAssets
    }

    input_df = pd.DataFrame([applicant_data])
    print("Converted DataFrame:\n", input_df)
    return input_df

