import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import pandas as pd
from convert_csv import ydre_vandstande, indre_vandstande, wather_data
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

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


df_indre = pd.DataFrame(indre_vandstande, columns=["Data1"])
df_ydre = pd.DataFrame(ydre_vandstande, columns=["Data2"])
df_weather = pd.DataFrame(wather_data, columns=["Wind Direction", "Wind Speed", "Gust Wind"])

# Step 1: Find the minimum length between the arrays
min_length = min(len(df_indre), len(df_ydre), len(df_weather))

# Step 2: Truncate all dataframes to the minimum length
df_indre_truncated = df_indre.iloc[:min_length]
df_ydre_truncated = df_ydre.iloc[:min_length]
df_weather_truncated = df_weather.iloc[:min_length]

# Step 3: Interpolate the smaller data arrays to match the length of the larger array (df_indre)
df2_interpolated = pd.DataFrame(
    np.interp(
        np.linspace(0, len(df_ydre_truncated) - 1, len(df_indre_truncated)),  # Target positions in df_indre size
        np.arange(len(df_ydre_truncated)),  # Original positions in df_ydre size
        df_ydre_truncated["Data2"]  # Data values from df_ydre
    ),
    columns=["Data2"]
)

# Step 4: Interpolate weather data for each column to match the truncated array length
interpolated_weather_values = []
for col in df_weather_truncated.columns:
    interpolated_weather_values.append(np.interp(
        np.linspace(0, len(df_weather_truncated) - 1, len(df_indre_truncated)),
        np.arange(len(df_weather_truncated)),
        df_weather_truncated[col]
    ))

# Create a DataFrame for the interpolated weather data
df_weather_interpolated = pd.DataFrame(np.column_stack(interpolated_weather_values), columns=df_weather.columns)

# Step 5: Combine both water heights data and weather data (as needed)
df_water_interpolated = pd.concat([df_indre_truncated, df2_interpolated], axis=1)

# Step 6: Convert to numpy array (final output for water heights)
water_heights_array = df_water_interpolated.to_numpy()
weather_data_array = df_weather_interpolated.to_numpy()
# Output the final water heights array and the interpolated weather data

# print(water_heights_array)
# print(weather_data_array)


# Define the weather and water conditions
wether = np.array(weather_data_array) # targets: [wether_directions, wether_wind_speeds, gust_wind] for each weather condition
water_heights_array = np.array(water_heights_array) # Targets: [water_height_1, water_height_2] for each weather condition

water_heights_array = interpolate_nan(water_heights_array)

contains_nan1 = np.isnan(water_heights_array).any()
contains_nan2 = np.isnan(wether).any()
print("Contains NaN:", contains_nan1, contains_nan2)

print(wether)
print(water_heights_array)

# Define the model
model = models.Sequential([
    layers.Input(shape=(3,)),  # Specify input shape explicitly
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(2)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error') # try NAdam optimizer and thers https://www.geeksforgeeks.org/optimizers-in-tensorflow/
# Train the model
model.fit(wether, water_heights_array, epochs=1, verbose=1)

# Test the model with a new weather condition
test_weather = np.array([[164.6, 4, 4.5], [164.6, 4, 4.5], [200, 10, 12]])  # Example: Predict water heights for weather conditions [wind direction, wind speed, gust wind]
predicted_heights = model.predict(test_weather)

for i, prediction in enumerate(predicted_heights):
    print(f"Test Sample {i + 1}:")
    print(f"Predicted water height 1: {prediction[0]:.2f}")
    print(f"Predicted water height 2: {prediction[1]:.2f}")