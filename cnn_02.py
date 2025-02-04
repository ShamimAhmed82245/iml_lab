import pandas as pd
import numpy as np
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping  # For early stopping
import matplotlib.pyplot as plt

# Generate sample timestamps (assuming new start date)
start_date = datetime.datetime(2020, 1, 1)  # Adjust as needed
end_date = datetime.datetime(2023, 12, 31)
date_range = pd.date_range(start_date, end_date, freq='H')

# Generate random temperature values
temperatures = np.random.randint(10, 30, size=len(date_range))

# Create the DataFrame
df = pd.DataFrame({'Timestamp': date_range, 'Temperature': temperatures})
df.set_index('Timestamp', inplace=True)

# Preprocess the data
def preprocess_data(df, look_back=24):
    X, y = [], []
    for i in range(len(df) - look_back):
        X.append(df.iloc[i:(i+look_back), 0].values)
        y.append(df.iloc[i+look_back, 0])
    return np.array(X), np.array(y)

# Reshape the data for CNN input
look_back = 24
X, y = preprocess_data(df, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# **Potentially Adjust Batch Size:**
# Consider adjusting the batch size based on your dataset size and hardware.
# A larger dataset or limited memory might benefit from a smaller batch size.
batch_size = 32  # Adjust as needed

# Create the CNN model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(look_back, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5)  # Monitor validation loss

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=batch_size,
                    validation_data=(X_test, y_test), verbose=2, callbacks=[early_stopping])

# Make predictions
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)

# Evaluate the model
# Calculate RMSE and MAPE
print('Train RMSE:', np.sqrt(np.mean((trainPredict[:,0] - y_train)**2)))
print('Test RMSE:', np.sqrt(np.mean((testPredict[:,0] - y_test)**2)))
# ...other evaluation metrics

# Plot the learning curve
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
