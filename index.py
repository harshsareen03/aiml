import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
# df =pd.read_csv('data.csv')
# print(df.head())
# df.head()
# print(df.tail())
# print(df.shape)
# print(df.info())
# print(df.describe())

df=pd.read_csv('data.csv')


print(df.isnull().sum())

df['age']=df['age'].fillna(df['age'].mean())

df['age']=df['age'].fillna(df['age'].median())

df['joined']=df['joined'].fillna('2020-01-01')

print(df.isnull().sum())


df[df['age']>120]
df.loc[df['age']>120, 'age']=df['age'].mean()

# df.loc[df['purchase']]=df['purchase'].map({'yes':1, 'no':0})

print(df.head())








# feature engineering technique

# df['age_group']=pd.cut(df['age'],bins=[0,18,35,60,120], labels=['child','youth','adult','senior']  )
df['gender']=df['gender'].map({'male':0,"female":1})

df['income_per_age']=df['salary']/df['age']

print(df.head())


# text featuring engineering

# 1)polynomial features

poly=PolynomialFeatures(degree=2, include_bias=False)
X_poly=poly.fit_transform(df[['age','salary']])
print(X_poly[:5])