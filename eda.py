from load_data import load_and_preprocess
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. Load preprocessed data
df = load_and_preprocess("credit_dataset.csv")

"""
print(df.info())
print(df.describe())
print(df.shape)
print(df.isnull().sum())
"""


def find_outlters(df, columns):
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1

    outliers = df[(df[columns] < Q1 - 1.5*IQR) | 
                (df[columns] > Q3 + 1.5*IQR)]
    
    return len(outliers)


def cap_outliers(df, columns):
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[columns] = np.where(df[columns] < lower_bound, lower_bound, df[columns])
    df[columns] = np.where(df[columns] > upper_bound, upper_bound, df[columns])
    
    return df



def Visualize_distribution(df, columns):
    # 2. Visualize the distribution of a numerical column
    plt.figure(figsize=(20,12))
    plt.hist(df[columns], bins=50, color='blue', edgecolor='black')
    plt.title(f'Distribution of {columns}')
    plt.xlabel(columns)
    plt.ylabel('Frequency')
    plt.show()


def Visualize_countplot(df, columns):
    # 3. Boxplot to identify outliers in a numerical column
    plt.figure(figsize=(40,6))
    sns.boxplot(x=df[columns])
    plt.title(f'Boxplot of {columns}')
    plt.show()



#adjusting dataset for analysis by creating new features and removing unneccessary ones by heatmap
df['household_size']= df['CNT_FAM_MEMBERS']


df['age']= df['DAYS_BIRTH'] / 365
df['age']=df['age'].astype(int)

df = df.drop(columns=['DAYS_BIRTH','CNT_FAM_MEMBERS','CNT_CHILDREN','FLAG_MOBIL','FLAG_PHONE','FLAG_EMAIL','FLAG_WORK_PHONE'])
df=df.dropna()


# change to num values for better visualization
numaric_cols= df.select_dtypes(include=['int64', 'float64'])


# check outliers for numerical columns
for col in numaric_cols:
    if find_outlters(df, col) !=0:
        print( col,find_outlters(df, col))
print ("--------------------------------------------------")

# cap outliers for numerical columns
df['AMT_INCOME_TOTAL_log'] = np.log1p(df['AMT_INCOME_TOTAL'])
df=df.drop(columns=['AMT_INCOME_TOTAL'])
df.dropna()

cap_outliers(df,'DAYS_EMPLOYED')

cap_outliers(df,'household_size')
cap_outliers(df,'AMT_INCOME_TOTAL_log')


numaric_cols2= df.select_dtypes(include=['int64', 'float64'])

# check outliers for numerical columns after capping
count=0
for col in numaric_cols2:
    if find_outlters(df, col) !=0:
        count=+1
        print( col,find_outlters(df, col))
if count==0:
    print("No outlters found")
print ("--------------------------------------------------")



# 4. Correlation heatmap for numerical features
numaric_cols_3= df.select_dtypes(include=['int64', 'float64'])
coo_matrix= numaric_cols_3.corr()

plt.figure(figsize=(12,10))
sns.heatmap(coo_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, linecolor='black', cbar=True)
plt.title('Correlation Matrix')
plt.show()








# Correlation Matrix split into two stages for better readability

# Stage 1: print first half of the columns
print("=== Correlation Matrix Part 1 ===")
print(coo_matrix.iloc[:, :len(coo_matrix)//2])

# Stage 2: print second half of the columns
print("\n=== Correlation Matrix Part 2 ===")
print(coo_matrix.iloc[:, len(coo_matrix)//2:])


