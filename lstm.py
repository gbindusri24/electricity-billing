#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


st.title("Forecasting Apple Dataset Using LSTM :")

# Load your dataset
Apple = pd.read_csv(r'C:\Users\SHYAM SASHANK\OneDrive\Desktop\Apple_dataset.csv')
columns_to_drop1 = ["Open", "High", "Low", "Adj Close", "Volume"]
df = Apple.drop(columns=columns_to_drop1)
df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

# Handle missing values
df.dropna(inplace=True)

# Normalize the data
scaler = MinMaxScaler()
df['y'] = scaler.fit_transform(df[['y']])

# Function to prepare data for LSTM
def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

# Convert the time series data to supervised learning
time_steps = 10  # You can adjust this parameter based on your data
X, y = prepare_data(df[['y']].values, time_steps)

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=25, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=25, return_sequences=False))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=20, batch_size=64)


# Date input from the user
date_input = st.date_input("Enter a date for forecasting:", pd.to_datetime('today'))

# Prepare data for forecasting
last_data = df['y'].values[-time_steps:]
last_data_scaled = scaler.transform(last_data.reshape(-1, 1))
last_data_reshaped = np.reshape(last_data, (1, time_steps, 1))

# Forecast
forecast = model.predict(last_data_reshaped)
forecast = scaler.inverse_transform(forecast)[0, 0]

# Display the forecast
st.write('Forecast:')
st.write(forecast)




