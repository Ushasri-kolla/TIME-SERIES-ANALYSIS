# 📊 Time-Series Analysis: ARIMA, SARIMA, LSTM, PROPHET

A comprehensive guide and implementation of advanced time-series forecasting models including classical statistical approaches (ARIMA/SARIMA) and modern deep learning techniques (LSTM) with real-world applications.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## 📑 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🔧 Technologies Used](#-technologies-used)
- [📚 Models Covered](#-models-covered)
- [📁 Project Structure](#-project-structure)
- [🚀 Installation & Setup](#-installation--setup)
- [💻 Usage & Examples](#-usage--examples)
- [📈 Model Comparison](#-model-comparison)
- [🔍 Key Features](#-key-features)
- [📊 Results & Visualizations](#-results--visualizations)
- [👥 Collaborators](#-collaborators)
- [✨ Author](#-author)

## 🎯 Project Overview

This project demonstrates a comprehensive approach to time-series analysis and forecasting using multiple modeling techniques:

- **Classical Statistical Models**: ARIMA and SARIMA for understanding temporal dependencies
- **Deep Learning Models**: LSTM neural networks for capturing complex non-linear patterns
- **Modern Prophet Model**: Facebook's Prophet for business forecasting with seasonality
- **Real-world Data**: Stock price prediction and financial forecasting applications

### Why Time-Series Analysis?

⏰ Time-series data is everywhere:
  - 📈 Stock market prices
  - 🏥 Patient vital signs over time
  - 🌡️ Weather and temperature forecasts
  - 👥 Website traffic and user engagement
  - 💼 Sales and revenue trends

## 🔧 Technologies Used

```
📦 Core Libraries:
  • pandas: Data manipulation and time-series handling
  • numpy: Numerical computations
  • scikit-learn: Machine learning preprocessing
  • statsmodels: ARIMA and SARIMA models
  • fbprophet: Facebook's Prophet for forecasting
  • tensorflow/keras: Deep learning (LSTM)

📊 Visualization:
  • matplotlib: Static plotting
  • seaborn: Statistical data visualization
  • plotly: Interactive visualizations

🗄️ Data Management:
  • CSV data handling
  • Time-series indexing and resampling
```

## 📚 Models Covered

### 1. 📊 ARIMA (AutoRegressive Integrated Moving Average)
- **Best For**: Univariate stationary or differenced time-series
- **Parameters**: (p, d, q)
  - p: AutoRegressive component
  - d: Differencing order
  - q: Moving Average component
- **Advantages**: Interpretable, fast, works with limited data
- **Limitations**: Assumes linear relationships, requires stationarity

### 2. 🔄 SARIMA (Seasonal ARIMA)
- **Best For**: Time-series with seasonal patterns
- **Parameters**: (p,d,q) × (P,D,Q,s) - seasonal extensions
- **Advantages**: Captures seasonal patterns, scalable to longer seasonality
- **Limitations**: Computationally expensive, requires tuning multiple parameters

### 3. 🧠 LSTM (Long Short-Term Memory)
- **Best For**: Complex non-linear patterns, multiple features
- **Architecture**: Recurrent neural network with memory cells
- **Advantages**: Handles long-term dependencies, multivariate data
- **Limitations**: Requires large datasets, black-box model

### 4. 🚀 Prophet
- **Best For**: Business forecasting with clear seasonality
- **Features**: Automatic changepoint detection, holiday effects
- **Advantages**: Robust to missing data, interpretable components
- **Limitations**: Less accurate for short-term forecasts, assumes trends

## 📁 Project Structure

```
TIME-SERIES-ANALYSIS/
├── app.py                    # Main application/demo
├── stock-price.py            # Stock price forecasting
├── data.csv                  # Sample time-series dataset
├── README.md                 # This file
└── results/                  # Output visualizations
    ├── arima_forecast.png
    ├── lstm_comparison.png
    └── prophet_decomposition.png
```

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip package manager
```

### Install Dependencies
```bash
# Clone the repository
git clone https://github.com/Ushasri-kolla/TIME-SERIES-ANALYSIS.git
cd TIME-SERIES-ANALYSIS

# Install required packages
pip install pandas numpy scikit-learn statsmodels fbprophet tensorflow matplotlib seaborn plotly

# Or use requirements file (if available)
pip install -r requirements.txt
```

## 💻 Usage & Examples

### 1. ARIMA Forecasting
```python
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# Load data
df = pd.read_csv('data.csv', parse_dates=['date'], index_col='date')

# Fit ARIMA model
model = ARIMA(df['price'], order=(1, 1, 1))
results = model.fit()

# Forecast
forecast = results.get_forecast(steps=30)
print(forecast.summary_frame())
```

### 2. SARIMA with Seasonality
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMA(1,1,1)×(1,1,1,12) for monthly seasonality
model = SARIMAX(df['price'], 
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12))
results = model.fit()
forecast = results.get_forecast(steps=12)
```

### 3. LSTM Deep Learning
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(lookback, 1)),
    Dense(25, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Predict
predictions = model.predict(X_test)
```

### 4. Facebook Prophet
```python
from fbprophet import Prophet

# Prepare data with 'ds' and 'y' columns
df_prophet = df.reset_index()
df_prophet.columns = ['ds', 'y']

# Fit Prophet model
model = Prophet(yearly_seasonality=True, daily_seasonality=False)
model.fit(df_prophet)

# Make forecast
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
model.plot(forecast)
```

## 📈 Model Comparison

| Model | Speed | Accuracy | Seasonality | Non-linearity | Data Required |
|-------|-------|----------|-------------|---------------|---------------|
| **ARIMA** | ⚡⚡⚡ | ⭐⭐⭐ | ❌ | ❌ | 📉 Low |
| **SARIMA** | ⚡⚡ | ⭐⭐⭐⭐ | ✅ | ❌ | 📊 Medium |
| **LSTM** | ⚡ | ⭐⭐⭐⭐⭐ | ✅ | ✅ | 📈 High |
| **Prophet** | ⚡⚡ | ⭐⭐⭐⭐ | ✅ | ⚠️ | 📊 Medium |

## 🔍 Key Features

✨ **Data Preprocessing**
  - Time-series indexing and resampling
  - Stationarity testing (ADF test)
  - Differencing and transformations
  - Missing value handling

✨ **Model Evaluation**
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)
  - ACF/PACF analysis

✨ **Visualization**
  - Original vs Forecast comparison
  - Residual analysis
  - Seasonal decomposition
  - Confidence intervals

✨ **Real-world Applications**
  - Stock price prediction
  - Revenue forecasting
  - Demand planning
  - Anomaly detection

## 📊 Results & Visualizations

### Performance Metrics Example
```
Model Performance on Test Set:

ARIMA:
  MAE:  15.32
  RMSE: 18.95
  MAPE: 2.15%

SARIMA:
  MAE:  12.45
  RMSE: 15.67
  MAPE: 1.78%

LSTM:
  MAE:  8.23
  RMSE: 10.45
  MAPE: 1.12%

Prophet:
  MAE:  11.89
  RMSE: 14.32
  MAPE: 1.69%
```

## 👥 Collaborators

**Chetan29-30** (Chetankumar Ganesh Mete)
- 🔗 GitHub: [@Chetan29-30](https://github.com/Chetan29-30)
- 💼 Role: Co-Developer
- 🎯 Contributions: LSTM implementation, data preprocessing, and model comparison framework

## ✨ Author

**Ushasri Kolla**
- 🔗 GitHub: [@Ushasri-kolla](https://github.com/Ushasri-kolla)
- 📧 Contact: [GitHub Profile](https://github.com/Ushasri-kolla)

## 📚 Learning Resources

📖 **Recommended Reading**:
  - "Forecasting: Principles and Practice" - Rob Hyndman
  - ARIMA/SARIMA: statsmodels documentation
  - LSTM: TensorFlow/Keras documentation
  - Prophet: Facebook Research papers

## 🎓 Key Concepts

1. **Stationarity**: Essential for ARIMA models
2. **Autocorrelation**: Understanding temporal dependencies
3. **Seasonality**: Periodic patterns in data
4. **Trend**: Long-term direction of time-series
5. **Residuals**: Model error analysis

## 🚀 Future Enhancements

- [ ] Ensemble methods combining multiple models
- [ ] Real-time forecasting dashboard
- [ ] AutoML for automatic model selection
- [ ] Multi-step ahead forecasting
- [ ] Transfer learning for related datasets
- [ ] GPU acceleration for LSTM training
- [ ] Model explainability (SHAP values)
- [ ] Production-ready API deployment

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- 📊 Statsmodels team for ARIMA/SARIMA implementations
- 🔬 Facebook Research for Prophet
- 🧠 TensorFlow/Keras team for deep learning tools
- 📈 Data science community for best practices

---

⭐ **If this project helps you, please consider giving it a star!** ⭐

💡 **Questions or suggestions?** Open an issue on GitHub!

🚀 **Happy Forecasting!**
