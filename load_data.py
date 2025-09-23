import pandas as pd

df=pd.read_csv("credit_dataset.csv")

print(df.head)

print("Dataset shape:",df.shape)



print(df.dtypes)

df=df.drop(columns=['ID'])
df['OCCUPATION_TYPE']=df['OCCUPATION_TYPE'].fillna('Unknown')
print(df.isnull().sum())

