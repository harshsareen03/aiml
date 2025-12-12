from sklearn.preprocessing import MinMaxScaler
import pandas as pd

scaler=MinMaxScaler()
df=pd.read_csv('sp.csv')

df[['Age','salary']]=scaler.fit_transform(df[['Age','salary']])

print(df.head())