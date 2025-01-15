import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from convert_csv import indre_vandstande, ydre_vandstande

def normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

def denormalize(data, min_val, max_val):
    return data * (max_val - min_val) + min_val

# Example arrays (input format)
indre_måling = indre_vandstande
ydre_måling = ydre_vandstande
# Step 1: Convert to DataFrame
# Convert the raw lists into pandas DataFrames for easier handling
indre_df = pd.DataFrame(indre_måling, columns=["timestamp", "water_level_indre"])
ydre_df = pd.DataFrame(ydre_måling, columns=["timestamp", "water_level_ydre"])

# Convert timestamps from Unix epoch to human-readable datetime format
indre_df["timestamp"] = pd.to_datetime(indre_df["timestamp"], unit="s")
ydre_df["timestamp"] = pd.to_datetime(ydre_df["timestamp"], unit="s")

# Step 1.1: Remove duplicates by aggregating water levels
# Group by timestamp and take the mean water level for duplicates
indre_df = indre_df.groupby("timestamp", as_index=False).mean()
ydre_df = ydre_df.groupby("timestamp", as_index=False).mean()

# Set the timestamp as the index for resampling purposes
indre_df.set_index("timestamp", inplace=True)
ydre_df.set_index("timestamp", inplace=True)

# Step 2: Resample to a common time interval (e.g., every 10 minutes)
# Interpolate missing values to handle differing intervals
indre_resampled = indre_df.resample("10T").interpolate()
ydre_resampled = ydre_df.resample("10T").interpolate()

# Step 3: Merge the two datasets on timestamps
# Align datasets on the same timeline
merged_df = pd.merge(indre_resampled, ydre_resampled, left_index=True, right_index=True, how="inner")
merged_df.reset_index(inplace=True)

# Step 4: Normalize the water level measurements
min_level = merged_df[["water_level_indre", "water_level_ydre"]].min().min()
max_level = merged_df[["water_level_indre", "water_level_ydre"]].max().max()

# Normalize to the range [0, 1]
merged_df["water_level_indre"] = normalize(merged_df["water_level_indre"], min_level, max_level)
merged_df["water_level_ydre"] = normalize(merged_df["water_level_ydre"], min_level, max_level)

# Step 5: Create sliding windows for time series input
def create_sliding_windows(data, window_size, prediction_size):
    X, y = [], []
    for i in range(len(data) - window_size - prediction_size + 1):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size : i + window_size + prediction_size])
    return np.array(X), np.array(y)

window_size = 6
prediction_size = 1

X, y = create_sliding_windows(
    merged_df[["water_level_indre", "water_level_ydre"]].values, window_size, prediction_size
)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Step 6: Build and train the TensorFlow model
model = Sequential([
    LSTM(64, activation="relu", input_shape=(window_size, 2)),
    Dense(32, activation="relu"),
    Dense(prediction_size * 2)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(
    X_train, y_train.reshape(-1, prediction_size * 2),
    validation_data=(X_test, y_test.reshape(-1, prediction_size * 2)),
    epochs=3,
    batch_size=32
)

# Step 7: Evaluate and Predict
loss, mae = model.evaluate(X_test, y_test.reshape(-1, prediction_size * 2))
print(f"Test Loss: {loss}, Test MAE: {mae}")

predictions = model.predict(X_test).reshape(-1, prediction_size, 2)
predictions_denorm = denormalize(predictions, min_level, max_level)
for i, pred in enumerate(predictions[:5]):
    print(f"Prediction {i+1}:")
    print(f"  Indre: {pred[0][0]:.4f}")
    print(f"  Ydre: {pred[0][1]:.4f}")