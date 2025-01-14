import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam, Nadam
import pandas as pd
from old_csv_convert import ydre_vandstande, indre_vandstande, wather_data
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Disable parallelism for TensorFlow
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

# Function to interpolate NaN values in the dataset
def interpolate_nan(array):
    for col in range(array.shape[1]):
        valid_indices = ~np.isnan(array[:, col])
        if valid_indices.any():
            interp_func = interp1d(
                np.where(valid_indices)[0], array[valid_indices, col], 
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )
            array[:, col] = interp_func(np.arange(len(array)))
    return array

# Step 1: Load the data
df_indre = pd.DataFrame(indre_vandstande, columns=["Data1"])
df_ydre = pd.DataFrame(ydre_vandstande, columns=["Data2"])
df_weather = pd.DataFrame(wather_data, columns=["Wind Direction", "Wind Speed", "Gust Wind"])

# Step 2: Truncate data to match the minimum length
min_length = min(len(df_indre), len(df_ydre), len(df_weather))
df_indre_truncated = df_indre.iloc[:min_length]
df_ydre_truncated = df_ydre.iloc[:min_length]
df_weather_truncated = df_weather.iloc[:min_length]

# Step 3: Interpolate the data to match the same length
df2_interpolated = pd.DataFrame(
    np.interp(
        np.linspace(0, len(df_ydre_truncated) - 1, len(df_indre_truncated)),
        np.arange(len(df_ydre_truncated)),
        df_ydre_truncated["Data2"]
    ),
    columns=["Data2"]
)

# Interpolate weather data
interpolated_weather_values = []
for col in df_weather_truncated.columns:
    interpolated_weather_values.append(np.interp(
        np.linspace(0, len(df_weather_truncated) - 1, len(df_indre_truncated)),
        np.arange(len(df_weather_truncated)),
        df_weather_truncated[col]
    ))

# Create a DataFrame for the interpolated weather data
df_weather_interpolated = pd.DataFrame(np.column_stack(interpolated_weather_values), columns=df_weather.columns)

# Step 4: Combine water heights and weather data
df_water_interpolated = pd.concat([df_indre_truncated, df2_interpolated], axis=1)
water_heights_array = df_water_interpolated.to_numpy()
weather_data_array = df_weather_interpolated.to_numpy()

# Interpolate NaN values
water_heights_array = interpolate_nan(water_heights_array)

# Check if NaNs remain
contains_nan1 = np.isnan(water_heights_array).any()
contains_nan2 = np.isnan(weather_data_array).any()
print("Contains NaN:", contains_nan1, contains_nan2)

# Prepare the data for time series forecasting
def create_time_series_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

# Example: Use the last 10 time steps to predict the next time step
sequence_length = 10
X_weather, y_weather = create_time_series_sequences(weather_data_array, sequence_length)
X_water, y_water = create_time_series_sequences(water_heights_array, sequence_length)

# Define the model
model = models.Sequential([
    layers.Input(shape=(sequence_length, 3)),  # Input shape: sequence_length x 3 (weather conditions)
    layers.LSTM(64, activation='relu', return_sequences=False),  # LSTM layer
    layers.Dense(2)  # Output: Predict 2 water heights
])

# Compile the model
model.compile(optimizer=Nadam(learning_rate=0.001), loss='mean_squared_error')

# Step 5: Train the model
model.fit(X_weather, y_water, epochs=5, batch_size=32)

# Step 6: Evaluate the model
model.evaluate(X_weather, y_water)

def predict_multiple_steps(model, initial_input, n_steps):
    predictions = []
    input_seq = initial_input.copy()  # Start with the last known sequence
    
    # Ensure the input sequence has the correct shape (1, sequence_length, 3)
    input_seq = np.expand_dims(input_seq, axis=0)  # Add batch dimension

    print(f"Initial input shape: {input_seq.shape}")  # Debugging line
    
    for _ in range(n_steps):
        # Predict the next step
        pred = model.predict(input_seq)
        predictions.append(pred)
        
        # Debugging line to check the prediction shape
        print(f"Predicted shape: {pred.shape}, Prediction: {pred}")

        # Update the input sequence:
        # Shift left along the sequence dimension
        input_seq = np.roll(input_seq, -1, axis=1)

        # Append the predicted water heights and keep the wind data the same
        input_seq[0, -1, 0] = pred[0][0]  # Water height 1 (replace first feature)
        input_seq[0, -1, 1] = pred[0][1]  # Water height 2 (replace second feature)

        # Wind data (third feature) remains unchanged
        # input_seq[0, -1, 2] remains the same

    return np.array(predictions)



# Example: Predict the next 5 time steps
initial_input = X_weather[-1]  # Start from the last input sequence in the training data
predicted_heights = predict_multiple_steps(model, initial_input, 5)

print("Predicted future water heights:")
for i, prediction in enumerate(predicted_heights):
    print(f"Step {i + 1}:")
    print(f"Predicted water height 1: {prediction[0][0]:.2f}")
    print(f"Predicted water height 2: {prediction[0][1]:.2f}")