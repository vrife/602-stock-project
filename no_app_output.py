import joblib
from tensorflow.keras.models import load_model
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
from flask import Flask, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

lin_reg = joblib.load('models/lin_reg.joblib')
poly_reg = joblib.load('models/poly_reg.joblib')
default_lasso = joblib.load('models/default_lasso.joblib')
tuned_lasso = joblib.load('models/tuned_lasso.joblib')
default_ridge = joblib.load('models/default_ridge.joblib')
tuned_ridge = joblib.load('models/tuned_ridge.joblib')

# Load keras models
cnn_model = load_model('models/cnn')
lstm_model = load_model('models/lstm')

# Fetch stock data
stock_data = yf.Ticker("TSLA").history(period="max")

# Update models with new data
x = stock_data.index.values
y = stock_data['Close'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/5)

# For regression classifiers
last_date = x[-1].astype('int64') // 1e9
last_date_datetime = datetime.utcfromtimestamp(last_date)
next_date_datetime = last_date_datetime + timedelta(days=1)
next_date_timestamp = np.array([[pd.Timestamp(next_date_datetime).value]])
print(f"Date of the next stock price we'll be predicting: {next_date_datetime}")

# Fit and predict for Linear Regression
lin = PolynomialFeatures(degree=1, include_bias=False)
train_linear_features = lin.fit_transform(x_train.reshape(-1, 1))
next_date_timestamp = np.array([[pd.Timestamp(next_date_datetime).value]])
next_date_linear_features = lin.transform(next_date_timestamp)

lin_reg.fit(train_linear_features, y_train)
lin_pred = lin_reg.predict(next_date_linear_features)[0]

# Fit and predict for Polynomial Regression
poly = PolynomialFeatures(degree=16, include_bias=False)
train_poly_features = poly.fit_transform(x_train.reshape(-1, 1))
next_date_poly_features = poly.transform(next_date_timestamp)

poly_reg.fit(train_poly_features, y_train)
poly_pred = poly_reg.predict(next_date_poly_features)[0]

# Fit and predict for Default LASSO
default_lasso.fit(train_poly_features, y_train)
def_lasso_pred = default_lasso.predict(next_date_poly_features)[0]

# Fit and predict for Tuned LASSO
tuned_lasso.fit(train_poly_features, y_train)
tun_lasso_pred = tuned_lasso.predict(next_date_poly_features)[0]

# Fit and predict for Default Ridge
poly = PolynomialFeatures(degree=5, include_bias=False)
train_poly_features = poly.fit_transform(x_train.reshape(-1, 1))
next_date_ridge_features = poly.transform(next_date_timestamp)

default_ridge.fit(train_poly_features, y_train)
def_ridge_pred = default_ridge.predict(next_date_ridge_features)[0]

# Fit and predict for Tuned Ridge
tuned_ridge.fit(train_poly_features, y_train)
tun_ridge_pred = tuned_ridge.predict(next_date_ridge_features)[0]

# Note: For the neural network predictions, we often need to reshape the input data to match the model's expected input shape.
# Here, we are assuming that the neural networks take the last N days of prices to predict the next day.
# Thus, we use N=30 because that's how the model was trained(i.e., using the last 30 days of prices to predict the next day).

X_recent = y[-1:].reshape(1, 1, 1)  # Reshape to (samples, timesteps, features) for LSTM
cnn_pred = cnn_model.predict(X_recent)[0][0]

# For LSTM prediction
N = 30  # This is the offset, or the number of days of data the LSTM model expects as input
scaler = MinMaxScaler(feature_range=(0, 1))
last_N_days = y[-N:]
last_N_days_scaled = scaler.fit_transform(last_N_days.reshape(-1, 1))
X_recent_lstm = last_N_days_scaled.reshape(1, N, 1)
predicted_scaled_price = lstm_model.predict(X_recent_lstm)

# Inverse transform the prediction to get the actual stock price
lstm_pred = scaler.inverse_transform(predicted_scaled_price)[0][0]

print(f"lin_pred: ${round(lin_pred, 2)}")
print(f"poly_pred: ${round(poly_pred, 2)}")
print(f"def_lasso_pred: ${round(def_lasso_pred, 2)}")
print(f"tun_lasso_pred: ${round(tun_lasso_pred, 2)}")
print(f"def_ridge_pred: ${round(def_ridge_pred, 2)}")
print(f"tun_ridge_pred: ${round(tun_ridge_pred, 2)}")
print(f"cnn_pred: ${round(cnn_pred, 2)}")
print(f"lstm_pred: ${round(lstm_pred, 2)}")