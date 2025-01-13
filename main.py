import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras import layers, models # type: ignore
import pandas as pd
from convert_csv import ydre_vandstande, indre_vandstande
import matplotlib.pyplot as plt


# Example arrays
array1 = np.array(indre_vandstande)  # Larger array
array2 = np.array(ydre_vandstande)  # Smaller array

# Convert to pandas DataFrame for easier manipulation
df1 = pd.DataFrame(array1, columns=["Data1"])
df2 = pd.DataFrame(array2, columns=["Data2"])

# Step 1: Truncate the larger array if exact matching is required
min_length = min(len(df1), len(df2))
df1_truncated = df1.iloc[:min_length]
df2_truncated = df2.iloc[:min_length]

print("Truncated Matching Arrays:")
print(len(df1_truncated))
print(len(df2_truncated))

# Step 2: Interpolating smaller data to match larger array if continuity matters
df2_interpolated = pd.DataFrame(
    np.interp(
        np.linspace(0, len(df2) - 1, len(df1)),  # Target positions in df1 size
        np.arange(len(df2)),  # Original positions in df2 size
        df2["Data2"]  # Data values from df2
    ),
    columns=["Data2"]
)

# Save interpolated data to a text file
#df2_interpolated.to_csv("df2_interpolated.txt", index=False, header=True, sep='\t')

plt.figure(figsize=(10, 6))

# Plot Array1 (original larger data)
plt.plot(df1.index, df1["Data1"], 'o-', label="Array1", markersize=1)  # Points and curve

# Plot the interpolated version of Array2
plt.plot(df2_interpolated.index, df2_interpolated["Data2"], 'x-', label="Interpolated Array2", markersize=1)  # Points and curve

# Title and labels
plt.title("Point Plot with Curve of Array1 and Interpolated Array2")
plt.xlabel("Index")
plt.ylabel("Data Values")

# Show legend
plt.legend()

# Show grid
plt.grid(True)

# Save the plot as PNG
plt.savefig("array1_vs_interpolated_array2.png")

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
water_heights_array = np.array([
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
model.fit(wether, water_heights_array, epochs=100, verbose=0)

# Test the model with a new weather condition
test_weather = np.array([[0, 9.2, 12.31]])  # Example: Predict water heights for "sunny" weather (weather condition 0)
predicted_heights = model.predict(test_weather)

print(f"Predicted water height 1: {predicted_heights[0][0]:.2f}")
print(f"Predicted water height 2: {predicted_heights[0][1]:.2f}")
