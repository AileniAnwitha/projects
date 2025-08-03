import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load Data
df = yf.download("AAPL", start="2015-01-01", end="2023-12-31")
data = df[['Close']]
dataset = data.values

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset)

# Train-test split
train_len = int(len(dataset) * 0.8)
train_data = scaled_data[:train_len]
test_data = scaled_data[train_len - 60:]

# Create sequences
X_train, y_train = [], []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i])
    y_train.append(train_data[i])
X_train, y_train = np.array(X_train), np.array(y_train)

# Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
model.add(LSTM(50))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Test sequences
X_test = []
for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i])
X_test = np.array(X_test)

# Predict
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Compare
actual = dataset[train_len:]
rmse = np.sqrt(np.mean((predictions - actual[60:])**2))
print("LSTM RMSE:", rmse)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(actual[60:], label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.title("LSTM Stock Price Prediction")
plt.show()
