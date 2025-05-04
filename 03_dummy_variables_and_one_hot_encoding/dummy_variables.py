import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('homeprices.csv')

df_dummy = pd.get_dummies(df.town, dtype=int)
df = pd.concat([df,df_dummy], axis='columns')

X_train = df[['area','monroe township', 'robinsville']]
y_train = df[['price']]

model = LinearRegression()
model.fit(X_train,y_train)
pred = model.predict(X_train)
print(f'Predection: {pred}')

score = model.score(X_train,y_train)
print(f'Score: {score}')