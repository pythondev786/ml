import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df = pd.read_csv('homeprices.csv')

median = df.bedrooms.median()
df.bedrooms = df.bedrooms.fillna(median)

X_train = df[['area','bedrooms','age']]
y_train = df['price']

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict([[2500, 4, 5]])
print(f'prediction: {pred}')

print(model.coef_)
print(model.intercept_)
