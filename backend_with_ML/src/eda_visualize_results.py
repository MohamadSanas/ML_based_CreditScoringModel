# eda_visualize_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset with anomaly results
df_eligible = pd.read_csv("eligible_applicants.csv")
df_not_eligible = pd.read_csv("not_eligible_applicants.csv")


# Visualize income distribution vs eligibility
sns.boxplot(x="anomaly", y="AMT_INCOME_TOTAL_log", data=df_not_eligible)
plt.title("Income Distribution: Eligible vs Not Eligible")
plt.show()

# Example: visualize age difference
sns.boxplot(x="anomaly", y="age", data=df_not_eligible)
plt.title("Age Distribution: Eligible vs Not Eligible")
plt.show()