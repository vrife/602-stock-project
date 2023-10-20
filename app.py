# --- IMPORTS ---
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from tensorflow.keras.models import load_model

# --- MODEL LOADING ---
def load_models():
    return {
        'lin_reg': joblib.load('models/lin_reg.joblib'),
        'poly_reg': joblib.load('models/poly_reg.joblib'),
        'default_lasso': joblib.load('models/default_lasso.joblib'),
        'tuned_lasso': joblib.load('models/tuned_lasso.joblib'),
        'default_ridge': joblib.load('models/default_ridge.joblib'),
        'tuned_ridge': joblib.load('models/tuned_ridge.joblib'),
        'cnn_model': load_model('models/cnn'),
        'lstm_model': load_model('models/lstm')
    }

# --- DATA PREPARATION ---
def fetch_data():
    return yf.Ticker("TSLA").history(period="max")

def prepare_data(stock_data):
    x = stock_data.index.values
    y = stock_data['Close'].values
    return train_test_split(x, y, test_size=1/5)

# --- PREDICTION FUNCTIONS ---
def predict_regressor(model, y_train, train_features, next_date_features):
    
    model.fit(train_features, y_train)
    regressor_pred = model.predict(next_date_features)[0]

    return round(regressor_pred, 2)

def predict_cnn(model, y):

    X_recent = y[-1:].reshape(1, 1, 1)  # Reshape to (samples, timesteps, features) for LSTM
    cnn_pred = model.predict(X_recent)[0][0]

    return round(cnn_pred, 2)

def predict_lstm(model, y):

    N = 30  # This is the offset, or the number of days of data the LSTM model expects as input
    scaler = MinMaxScaler(feature_range=(0, 1))
    last_N_days = y[-N:]
    last_N_days_scaled = scaler.fit_transform(last_N_days.reshape(-1, 1))
    X_recent_lstm = last_N_days_scaled.reshape(1, N, 1)
    predicted_scaled_price = model.predict(X_recent_lstm)

    # Inverse transform the prediction to get the actual stock price
    lstm_pred = scaler.inverse_transform(predicted_scaled_price)[0][0]

    return round(lstm_pred, 2)

models = load_models()

def get_predictions():

    stock_data = fetch_data()
    x = stock_data.index.values
    y = stock_data['Close'].values
    x_train, _, y_train, _ = prepare_data(stock_data)

    # For regression classifiers
    last_date = x[-1].astype('int64') // 1e9
    last_date_datetime = datetime.utcfromtimestamp(last_date)
    next_date_datetime = last_date_datetime + timedelta(days=1)
    next_date_timestamp = np.array([[pd.Timestamp(next_date_datetime).value]])
    print(f"Date of the next stock price we'll be predicting: {next_date_datetime}")

    lin = PolynomialFeatures(degree=1, include_bias=False)
    train_linear_features = lin.fit_transform(x_train.reshape(-1, 1))
    next_date_linear_features = lin.transform(next_date_timestamp)

    poly = PolynomialFeatures(degree=16, include_bias=False)
    train_poly_features = poly.fit_transform(x_train.reshape(-1, 1))
    next_date_poly_features = poly.transform(next_date_timestamp)

    # Fit and predict for Default Ridge
    ridge = PolynomialFeatures(degree=5, include_bias=False)
    train_ridge_features = ridge.fit_transform(x_train.reshape(-1, 1))
    next_date_ridge_features = ridge.transform(next_date_timestamp)

    return {
        'predictions': {
            'Linear Regression': predict_regressor(models['lin_reg'], y_train, train_linear_features, next_date_linear_features),
            'Polynomial Regression': predict_regressor(models['poly_reg'], y_train, train_poly_features, next_date_poly_features),
            'Default LASSO': predict_regressor(models['default_lasso'], y_train, train_poly_features, next_date_poly_features),
            'Tuned LASSO': predict_regressor(models['tuned_lasso'], y_train, train_poly_features, next_date_poly_features),
            'Default Ridge': predict_regressor(models['default_ridge'], y_train, train_ridge_features, next_date_ridge_features),
            'Tuned Ridge': predict_regressor(models['tuned_ridge'], y_train, train_ridge_features, next_date_ridge_features),
            'Convolutional Neural Network': predict_cnn(models['cnn_model'], y),
            'LSTM': predict_lstm(models['lstm_model'], y)
            # ... call the functions for other models
        },
        'next_date': next_date_datetime.strftime("%m/%d/%Y")
    }

# --- FLASK SETUP ---
app = Flask(__name__)

@app.route('/')
def index():
    result = get_predictions()
    return render_template('index.html', predictions=result['predictions'], next_date=result['next_date'])

if __name__ == '__main__':
    app.run(debug=True)
