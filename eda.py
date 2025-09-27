from load_data import load_and_preprocess
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Load preprocessed data
df = load_and_preprocess("credit_dataset.csv")

"""
print(df.info())
print(df.describe())
print(df.shape)
print(df.isnull().sum())
"""

columns= 'AMT_INCOME_TOTAL'



# 2. Visualize the distribution of a numerical column
plt.figure(figsize=(10,6))
plt.hist(df[columns], bins=50, color='blue', edgecolor='black')
plt.title(f'Distribution of {columns}')
plt.xlabel(columns)
plt.ylabel('Frequency')
#plt.show()

# 3. Boxplot to identify outliers in a numerical column
plt.figure(figsize=(40,6))
sns.boxplot(x=df[columns])
plt.title(f'Boxplot of {columns}')
#plt.show()

# 4. Correlation heatmap for numerical features
numaric_cols= df.select_dtypes(include=['int64', 'float64'])
coo_matrix= numaric_cols.corr()

plt.figure(figsize=(20,15))
sns.heatmap(coo_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()






