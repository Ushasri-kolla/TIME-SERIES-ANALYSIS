import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.metrics import mean_squared_error
from math import sqrt

# Prophet
from prophet import Prophet

# ARIMA / SARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX # pyright: ignore[reportMissingImports]

# LSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =========================================
# Load Data
# =========================================
st.title("📈 Stock Market Forecast Dashboard")

data = pd.read_csv("data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

st.line_chart(data['Close'])

# =========================================
# Prophet Forecast
# =========================================
df = data.reset_index()[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
model_prophet = Prophet()
model_prophet.fit(df)
future = model_prophet.make_future_dataframe(periods=30)
forecast_prophet = model_prophet.predict(future)

rmse_prophet = sqrt(mean_squared_error(df['y'][-30:], forecast_prophet['yhat'][-30:]))

st.subheader("🔮 Prophet Forecast")
fig1 = model_prophet.plot(forecast_prophet)
st.pyplot(fig1)

# =========================================
# ARIMA Forecast
# =========================================
model_arima = ARIMA(data['Close'], order=(5,1,0))
model_fit_arima = model_arima.fit()
forecast_arima = model_fit_arima.forecast(steps=30)

rmse_arima = sqrt(mean_squared_error(data['Close'][-30:], forecast_arima))

st.subheader("📉 ARIMA Forecast")
fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.plot(data['Close'], label="Actual")
ax2.plot(pd.date_range(data.index[-1], periods=30, freq='D'), forecast_arima, label="Forecast", color="red")
ax2.legend()
st.pyplot(fig2)

# =========================================
# SARIMA Forecast
# =========================================
model_sarima = SARIMAX(data['Close'], order=(1,1,1), seasonal_order=(1,1,1,12))
model_fit_sarima = model_sarima.fit(disp=False)
forecast_sarima = model_fit_sarima.forecast(30)

rmse_sarima = sqrt(mean_squared_error(data['Close'][-30:], forecast_sarima))

st.subheader("📉 SARIMA Forecast")
fig3, ax3 = plt.subplots(figsize=(10,4))
ax3.plot(data['Close'], label="Actual")
ax3.plot(pd.date_range(data.index[-1], periods=30, freq='D'), forecast_sarima, label="Forecast", color="green")
ax3.legend()
st.pyplot(fig3)

# =========================================
# LSTM Forecast
# =========================================
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data[['Close']].values.reshape(-1,1))

train_size = int(len(scaled_data)*0.8)
train, test = scaled_data[:train_size], scaled_data[train_size:]

def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset)-time_step-1):
        X.append(dataset[i:(i+time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

X_train, y_train = create_dataset(train, 60)
X_test, y_test = create_dataset(test, 60)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model_lstm = Sequential()
model_lstm.add(LSTM(50, return_sequences=True, input_shape=(60,1)))
model_lstm.add(LSTM(50))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer="adam", loss="mean_squared_error")
model_lstm.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

train_predict = model_lstm.predict(X_train)
test_predict = model_lstm.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict  = scaler.inverse_transform(test_predict)

rmse_lstm = sqrt(mean_squared_error(scaler.inverse_transform(y_test.reshape(-1,1)), test_predict[:len(y_test)]))

st.subheader("🤖 LSTM Forecast")
fig4, ax4 = plt.subplots(figsize=(10,4))
ax4.plot(data.index, data['Close'], label="Actual")
ax4.plot(data.index[60:len(train_predict)+60], train_predict, label="Train Predict")
ax4.plot(data.index[len(train_predict)+(60*2)+1:len(data)-1], test_predict, label="Test Predict")
ax4.legend()
st.pyplot(fig4)

# =========================================
# Results
# =========================================
st.subheader("📊 Model Accuracy (RMSE)")
st.write(f"Prophet RMSE: {rmse_prophet:.2f}")
st.write(f"ARIMA RMSE:   {rmse_arima:.2f}")
st.write(f"SARIMA RMSE:  {rmse_sarima:.2f}")
st.write(f"LSTM RMSE:    {rmse_lstm:.2f}")
