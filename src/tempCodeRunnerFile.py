import pandas as pd


df = pd.read_csv("data/preprocessed_data.csv")
loan_approved = df[df["loan_status"]==1]

loan_approved.to_csv("data/aproved_loan.csv",index=False)




