import pickle

#load saved model
with open('../saved_models/model_home_price_prediction', 'rb') as f:
  model = pickle.load(f)

pred = model.predict([[7100]])
print(f'Prediction: {pred}')