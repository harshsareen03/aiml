from sklearn.model_selection import train_test_split

import pandas as pd


df=pd.read_csv('sp.csv')
x=df[['Age','salary']]


df=pd.read_csv('sp.csv')

y=df['purchased']
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=42)
print("X_train:\n", x_train)
print("X_test:\n", x_test)
