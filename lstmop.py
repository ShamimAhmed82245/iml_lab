import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('AirPassengers.csv')
data = data[['#Passengers']]  # Use the correct column name

# Step 2: Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_sequences(data, sequence_length):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(x), np.array(y)

sequence_length = 50
X, y = create_sequences(scaled_data, sequence_length)
X = np.expand_dims(X, axis=-1)  # Add feature dimension for LSTM
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Function to build the LSTM model
def build_model(units, learning_rate, optimizer, activation, dropout_rate):
    model = Sequential([
        Input(shape=(sequence_length, 1)),  # Add an Input layer
        LSTM(units, activation=activation, return_sequences=False),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer=optimizer(learning_rate=learning_rate), loss='mse')                                                     
    return model

model = build_model(32, 0.001, Adam, 'tanh', 0.02)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))


