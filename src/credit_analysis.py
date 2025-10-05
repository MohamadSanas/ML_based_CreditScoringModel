import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


eligible_df = pd.read_csv("data/eligible_applicants.csv")
not_eligible_df = pd.read_csv("data/not_eligible_applicants.csv")

eligible_df['aligibility'] = 'Eligible'
not_eligible_df['aligibility'] = 'Not Eligible'

df = pd.concat([eligible_df,not_eligible_df],ignore_index=True)

sns.set(style="whitegrid",palette="muted",font_scale=1)



# Visualize key features vs eligibility 

# 1. Age vs Income
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x="age", y="AMT_INCOME_TOTAL_log", hue="aligibility", alpha=0.6)
plt.title("Income vs Age: Eligible vs Not Eligible")
plt.xlabel("Age")
plt.ylabel("Log of Total Income")
plt.show()
"""


# 2. Car Ownership vs Eligibility
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='FLAG_OWN_CAR', hue='aligibility')
plt.title("Car Ownership: Eligible vs Not Eligible")
plt.xlabel("Owns Car (1=Yes, 0=No)")   
plt.ylabel("Count")
plt.show() 



plt.figure(figsize=(6,4))
sns.countplot(data=df, x='FLAG_OWN_REALTY', hue='aligibility')
plt.title("Property Ownership vs Eligibility")
plt.xlabel("Owns Property (0=No, 1=Yes)")
plt.ylabel("Count")
plt.show()
"""


plt.figure(figsize=(8,6))
sns.countplot(data=df, x='CODE_GENDER', hue='aligibility')
plt.title("Count of Applicants by Gender and Eligibility")
plt.xlabel("Gender (0 = Female, 1 = Male)")
plt.ylabel("Count")
plt.legend(title="Eligibility", labels=["Not Eligible", "Eligible"])
plt.show()




key_features = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'age', 
                'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL_log', 'household_size']

plt.figure(figsize=(10,8))
sns.heatmap(df[key_features].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap of Key Features")
plt.show()



