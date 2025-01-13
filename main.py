import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras import layers, models # type: ignore

# Define the weather conditions
# targets: [wether_directions, wether_wind_speeds, gust_wind] for each weather condition
wether = np.array([
    [0, 9.2, 12.3],  # North
    [90, 8.1, 12],  # East
    [180, 9.3, 10.8],  # South
    [270, 8.7, 10.7],  # West
    [180, 8.4, 11.4],  # South
])

# Targets: [water_height_1, water_height_2] for each weather condition
water_heights = np.array([
    [1.0, 2.0],  # Sunny
    [2.5, 3.1],  # Rainy
    [1.8, 2.4],  # Sunny
    [3.0, 3.8],  # Cloudy
    [1.2, 1.8],  # Rainy
])

# Define the model
model = models.Sequential([
    layers.Input(shape=(3,)),  # Specify input shape explicitly
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(2)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
model.fit(wether, water_heights, epochs=100, verbose=0)

# Test the model with a new weather condition
test_weather = np.array([[0, 9.2, 12.31]])  # Example: Predict water heights for "sunny" weather (weather condition 0)
predicted_heights = model.predict(test_weather)

print(f"Predicted water height 1: {predicted_heights[0][0]:.2f}")
print(f"Predicted water height 2: {predicted_heights[0][1]:.2f}")
