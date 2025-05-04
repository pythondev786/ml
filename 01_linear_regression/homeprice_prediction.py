import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('homeprices.csv')

model = LinearRegression()
X_train = df.drop(['price'], axis='columns')
y_train = df.price
model.fit(X_train, y_train)

pred = model.predict(pd.DataFrame({'area':[2600, 3000]}))
print(f'prediction: {pred}')

print(f'Coefficient: {model.coef_}')
print(f'Intercept: {model.intercept_}')

pred = model.predict(pd.read_csv('areas.csv'))
print(f'prediction: {pred}')

# Saving the model
with open('../saved_models/home_price', 'wb') as f:
    pickle.dump(model, f)