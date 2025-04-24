import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Streamlit Page Config
st.set_page_config(page_title="Stock Price Forecast | SUZLON", page_icon="📈")

st.title("📈 Live SUZLON Stock Price Predictor")

# Sidebar: select date range
start_date = st.sidebar.date_input("Start Date", datetime(2024, 11, 1))
end_date = st.sidebar.date_input("End Date", datetime(2025, 4, 24))

if start_date >= end_date:
    st.error("End date must be after start date.")
    st.stop()

# Download data
st.subheader("Stock Price Data 📊")
data = yf.download("SUZLON.NS", start=start_date, end=end_date)
st.write(data.tail())

# Plot closing price
st.subheader("Close Price Trend 📉")
fig, ax = plt.subplots()
ax.plot(data['Close'], label='Close Price')
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.legend()
ax.grid()
st.pyplot(fig)

# Data Preprocessing
data = data[['Close']]
dataset = data.values

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(dataset)

time_step = 10
x_train, y_train = [], []

for i in range(time_step, len(data_scaled)):
    x_train.append(data_scaled[i - time_step:i])
    y_train.append(data_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=0, shuffle=False, callbacks=[early_stop])

# Future Predictions
X_FUTURE = st.sidebar.number_input("Days to Predict", min_value=1, max_value=30, value=7)

test_data = data_scaled[-time_step:]
last_data = test_data.reshape(1, time_step, 1)

future_predictions = []

for _ in range(X_FUTURE):
    future_pred = model.predict(last_data)
    future_predictions.append(future_pred[0, 0])
    last_data = np.roll(last_data, shift=-1, axis=1)
    last_data[0, -1, 0] = future_pred

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Future Dates
curr_date = data.index[-1]
future_dates = [curr_date + timedelta(days=i+1) for i in range(X_FUTURE)]

# Create Future DataFrame
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_predictions.flatten()})
future_df.set_index('Date', inplace=True)

# Show future predictions
st.subheader(f"Future Predictions for Next {X_FUTURE} Days 📈")
st.dataframe(future_df)

# Plot future predictions
st.subheader("Prediction Graph 📊")
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(data.index, data['Close'], label='Actual Close Price', color='blue')
ax2.plot(future_df.index, future_df['Predicted Close'], label='Predicted Close Price', color='red', linestyle='dashed')
ax2.set_xlabel("Date")
ax2.set_ylabel("Close Price")
ax2.set_title("SUZLON Close Price Forecast")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

st.success("✅ Prediction completed!")
