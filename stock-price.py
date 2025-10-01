import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

# Prophet
from prophet import Prophet

# ARIMA / SARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# LSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ===============================
# 1. Load Data
# ===============================
data = pd.read_csv("data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

plt.figure(figsize=(10,4))
plt.plot(data['Close'])
plt.title("Stock Prices")
plt.show()

# ===============================
# 2. Prophet
# ===============================
df = data.reset_index()[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
model_prophet = Prophet()
model_prophet.fit(df)
future = model_prophet.make_future_dataframe(periods=30)
forecast_prophet = model_prophet.predict(future)

plt.figure(figsize=(10,4))
model_prophet.plot(forecast_prophet)
plt.title("Prophet Forecast")
plt.show()

rmse_prophet = sqrt(mean_squared_error(df['y'][-30:], forecast_prophet['yhat'][-30:]))

# ===============================
# 3. ARIMA
# ===============================
model_arima = ARIMA(data['Close'], order=(5,1,0))
model_fit_arima = model_arima.fit()
forecast_arima = model_fit_arima.forecast(steps=30)

plt.figure(figsize=(10,4))
plt.plot(data['Close'], label="Actual")
plt.plot(pd.date_range(data.index[-1], periods=30, freq='D'), forecast_arima, label="ARIMA Forecast", color="red")
plt.legend()
plt.title("ARIMA Forecast")
plt.show()

rmse_arima = sqrt(mean_squared_error(data['Close'][-30:], forecast_arima))

# ===============================
# 4. SARIMA
# ===============================
model_sarima = SARIMAX(data['Close'], order=(1,1,1), seasonal_order=(1,1,1,12))
model_fit_sarima = model_sarima.fit(disp=False)
forecast_sarima = model_fit_sarima.forecast(30)

plt.figure(figsize=(10,4))
plt.plot(data['Close'], label="Actual")
plt.plot(pd.date_range(data.index[-1], periods=30, freq='D'), forecast_sarima, label="SARIMA Forecast", color="green")
plt.legend()
plt.title("SARIMA Forecast")
plt.show()

rmse_sarima = sqrt(mean_squared_error(data['Close'][-30:], forecast_sarima))

# ===============================
# 5. LSTM
# ===============================
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
model_lstm.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

train_predict = model_lstm.predict(X_train)
test_predict = model_lstm.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict  = scaler.inverse_transform(test_predict)

plt.figure(figsize=(10,4))
plt.plot(data.index, data['Close'], label="Actual")
plt.plot(data.index[60:len(train_predict)+60], train_predict, label="Train Predict")
plt.plot(data.index[len(train_predict)+(60*2)+1:len(data)-1], test_predict, label="Test Predict")
plt.title("LSTM Forecast")
plt.legend()
plt.show()

rmse_lstm = sqrt(mean_squared_error(scaler.inverse_transform(y_test.reshape(-1,1)), test_predict[:len(y_test)]))

# ===============================
# 6. Results
# ===============================
print("Model RMSEs:")
print(f"Prophet RMSE: {rmse_prophet:.2f}")
print(f"ARIMA RMSE:   {rmse_arima:.2f}")
print(f"SARIMA RMSE:  {rmse_sarima:.2f}")
print(f"LSTM RMSE:    {rmse_lstm:.2f}")
