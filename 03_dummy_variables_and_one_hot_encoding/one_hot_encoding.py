import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.read_csv('homeprices.csv')
le = LabelEncoder()
df.town = le.fit_transform(df.town)

ohe = OneHotEncoder(sparse_output=False, dtype=int)
encoded_array = ohe.fit_transform(df[['town']])
df_encoded = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out(['town']))

X_train = pd.concat([df_encoded, df[['area']]], axis='columns')
X_train = X_train.drop(['town_0'],axis='columns')
y_train = df.price

model = LinearRegression()
model.fit(X_train,y_train)
pred = model.predict(X_train)
print(f'Prediction: {pred}')

score = model.score(X_train,y_train)
print(f'Score: {score}')