import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D

# Load the CSV file
file_path = 'sales-of-shampoo-over-a-three-year-period.csv'  # Update with your actual file path
data = pd.read_csv(file_path)

# Convert 'Month' to datetime
# Adjust the year assignment based on your data
years = [2021, 2022, 2023]
data['Year'] = (data.index // 12) + 2021  # Assuming 12 months per year
data['Month'] = pd.to_datetime(data['Month'] + '-' + data['Year'].astype(str), format='%d-%b-%Y')

# Sort by date
data = data.sort_values('Month').reset_index(drop=True)

# Extract sales data
sales_data = data['Sales of shampoo over a three year period'].dropna().values

# Define the number of time steps
n_steps = 3

# Function to split the sequence
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Split the sales data
X, y = split_sequence(sales_data, n_steps)

# Reshape for CNN
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# Define the CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=200, verbose=0)

# Create a date to index mapping
date_to_index = {date: idx for idx, date in enumerate(data['Month'])}

# Prediction functions
def get_index_from_date(target_date):
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date, format='%Y-%m-%d')
    if target_date in date_to_index:
        return date_to_index[target_date]
    else:
        raise ValueError("Date not found in the dataset.")

def predict_sales_for_date(target_date):
    try:
        target_date = pd.to_datetime(target_date, format='%Y-%m-%d')
    except ValueError:
        raise ValueError("Incorrect date format. Please use 'YYYY-MM-DD'.")
    
    target_idx = get_index_from_date(target_date)
    
    if target_idx < n_steps:
        raise ValueError(f"Not enough data to predict for {target_date.strftime('%Y-%m-%d')}.")
    
    input_indices = range(target_idx - n_steps, target_idx)
    input_sequence = sales_data[input_indices]
    x_input = input_sequence.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    return yhat[0][0]

def add_prediction_to_data(predicted_value, prediction_date):
    global sales_data, data, date_to_index
    sales_data = np.append(sales_data, predicted_value)
    data = data.append({'Month': prediction_date, 'Sales of shampoo over a three year period': predicted_value}, ignore_index=True)
    date_to_index[prediction_date] = len(sales_data) - 1

def predict_future_sales(start_date, periods):
    predictions = []
    current_date = pd.to_datetime(start_date, format='%Y-%m-%d')
    
    for _ in range(periods):
        next_date = current_date + pd.DateOffset(months=1)
        next_date_str = next_date.strftime('%Y-%m-%d')
        predicted_sales = predict_sales_for_date(next_date_str)
        predictions.append((next_date_str, predicted_sales))
        add_prediction_to_data(predicted_sales, next_date)
        current_date = next_date
    
    return predictions

# Example usage
last_date = data['Month'].iloc[-1]
next_month = last_date + pd.DateOffset(months=1)
next_month_str = next_month.strftime('%Y-%m-%d')

predicted_sales = predict_sales_for_date(next_month_str)
print(f"Predicted sales for {next_month_str}: {predicted_sales:.2f}")

# Predict next 3 months
future_predictions = predict_future_sales(next_month_str, 3)
for date, sales in future_predictions:
    print(f"Predicted sales for {date}: {sales:.2f}")
