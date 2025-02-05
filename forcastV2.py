from tensorflow.keras.metrics import RootMeanSquaredError # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, InputLayer # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from convert_csv import indre_vandstande, ydre_vandstande
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from tensorflow.keras.layers import Reshape

# Resample data to every 30 minutes
indre_df = indre_vandstande[0::6]
ydre_df = ydre_vandstande[0::3]

# Convert timestamps
indre_df.index = pd.to_datetime(indre_df['Timestamp'], format='%d-%m-%Y %H:%M')
ydre_df.index = pd.to_datetime(ydre_df['Timestamp'], format='%d-%m-%Y %H:%M')

# Extract water levels
indre_df = indre_df[['Water Level']]
ydre_df = ydre_df[['Water Level']]

# Normalize water levels
scaler_indre = MinMaxScaler()
scaler_ydre = MinMaxScaler()

indre_df['Water Level'] = scaler_indre.fit_transform(indre_df[['Water Level']])
ydre_df['Water Level'] = scaler_ydre.fit_transform(ydre_df[['Water Level']])

# Add timestamp-based sin/cos features
day = 60*60*24  # Seconds in a day
year = 365.2425 * day  # Seconds in a year

timestamps = indre_df.index.map(pd.Timestamp.timestamp)
indre_df['Day sin'] = np.sin(timestamps * (2 * np.pi / day))
indre_df['Day cos'] = np.cos(timestamps * (2 * np.pi / day))
indre_df['Year sin'] = np.sin(timestamps * (2 * np.pi / year))
indre_df['Year cos'] = np.cos(timestamps * (2 * np.pi / year))

timestamps = ydre_df.index.map(pd.Timestamp.timestamp)
ydre_df['Day sin'] = np.sin(timestamps * (2 * np.pi / day))
ydre_df['Day cos'] = np.cos(timestamps * (2 * np.pi / day))
ydre_df['Year sin'] = np.sin(timestamps * (2 * np.pi / year))
ydre_df['Year cos'] = np.cos(timestamps * (2 * np.pi / year))

# Convert DataFrame to LSTM format
def df_to_X_y(df, window_size, future_steps):
    df_as_np = df.to_numpy()
    X, y = [], []

    for i in range(len(df_as_np) - window_size - future_steps):
        X.append(df_as_np[i:i + window_size])
        y.append(df_as_np[i + window_size:i + window_size + future_steps])

    return np.array(X), np.array(y)

window_size = 10  # Past time steps
future_steps = 48  # Future time steps
num_features = 5 

indre_X, indre_y = df_to_X_y(indre_df, window_size, future_steps)
ydre_X, ydre_y = df_to_X_y(ydre_df, window_size, future_steps)

# Split into train/val/test sets
indre_X_train, indre_y_train = indre_X[:60000], indre_y[:60000]
indre_X_val, indre_y_val = indre_X[60000:70000], indre_y[60000:70000]
indre_X_test, indre_y_test = indre_X[100000:], indre_y[100000:]

ydre_X_train, ydre_y_train = ydre_X[:60000], ydre_y[:60000]
ydre_X_val, ydre_y_val = ydre_X[60000:70000], ydre_y[60000:70000]
ydre_X_test, ydre_y_test = ydre_X[100000:], ydre_y[100000:]

# Define LSTM Model
def build_model():
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(window_size, num_features)),
        LSTM(32, activation='relu', return_sequences=False),
        Dense(future_steps * num_features, activation='linear'),
        Reshape((future_steps, num_features))
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])
    return model

# Train Indre Model
indre_model = build_model()
indre_model_CP = ModelCheckpoint('Models/indre_modelV2.keras', save_best_only=True)
indre_model.fit(indre_X_train, indre_y_train, validation_data=(indre_X_val, indre_y_val), epochs=2, callbacks=[indre_model_CP])

# Train Ydre Model
ydre_model = build_model()
ydre_model_CP = ModelCheckpoint('Models/ydre_modelV2.keras', save_best_only=True)
ydre_model.fit(ydre_X_train, ydre_y_train, validation_data=(ydre_X_val, ydre_y_val), epochs=2, callbacks=[ydre_model_CP])

# Load best models
indre_model = load_model('Models/indre_modelV2.keras')
ydre_model = load_model('Models/ydre_modelV2.keras')

# Make Predictions
indre_future_predictions = indre_model.predict(indre_X_test[:1])
ydre_future_predictions = ydre_model.predict(ydre_X_test[:1])

# Inverse transform predictions to original scale
indre_future_predictions = scaler_indre.inverse_transform(indre_future_predictions.reshape(-1, 1)).flatten()
ydre_future_predictions = scaler_ydre.inverse_transform(ydre_future_predictions.reshape(-1, 1)).flatten()

# Make predictions
indre_input_data = indre_X_test[-1:]  # Use last test sample
ydre_input_data = ydre_X_test[-1:]

indre_future_predictions = indre_model.predict(indre_input_data)
ydre_future_predictions = ydre_model.predict(ydre_input_data)

indre_future_predictions = indre_future_predictions.reshape(-1, indre_future_predictions.shape[-1])  # Flatten time steps
ydre_future_predictions = ydre_future_predictions.reshape(-1, ydre_future_predictions.shape[-1]) 

# Plot each feature separately
for i in range(indre_future_predictions.shape[1]):  # Iterate over the number of features
    plt.plot(indre_future_predictions[:, i], label=f'Feature {i + 1}')

# Plot each feature separately
for i in range(ydre_future_predictions.shape[1]):  # Iterate over the number of features
    plt.plot(ydre_future_predictions[:, i], label=f'Feature {i + 1}')

# Plot the predictions
plt.figure(figsize=(14, 7))
plt.plot(indre_future_predictions, label='Indre Future Predictions', color='blue')
plt.plot(ydre_future_predictions, label='Ydre Future Predictions', color='red')
plt.xlabel('Time Steps')
plt.ylabel('Predicted Values')
plt.title('Indre vs Ydre Future Predictions')
plt.legend()
plt.tight_layout()
plt.savefig('PredictionsV2.png')
plt.show()
# plt.show()

"""
def predictions(indre_predictions_df, ydre_predictions_df, max_xticks=10):
    # Print column names for debugging
    print("Indre Predictions DataFrame columns:", indre_predictions_df.columns)
    print("Ydre Predictions DataFrame columns:", ydre_predictions_df.columns)
    
    # Set the 'Timestamp' column as the index for both DataFrames
    indre_predictions_df.set_index('Timestamp', inplace=True)
    ydre_predictions_df.set_index('Timestamp', inplace=True)
    
    # Create a figure
    plt.figure(figsize=(14, 7))
    
    # Plot both DataFrames on the same plot
    plt.plot(indre_predictions_df.index, indre_predictions_df.iloc[:, 0], label='Indre Predictions', color='blue')
    plt.plot(ydre_predictions_df.index, ydre_predictions_df.iloc[:, 0], label='Ydre Predictions', color='red')
    
    # Set labels and title
    plt.xlabel('Timestamp')
    plt.ylabel('Predicted Values')
    plt.title('Indre Predictions vs Ydre Predictions')
    plt.legend()
    
    # Limit the number of x-axis labels
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=max_xticks))
    
    # Show the plot
    plt.tight_layout()
    # plt.savefig('Predictions.png')
    plt.show()
"""

"""def predict_future(model, initial_window, window_size, steps_ahead, indre_df):
    predictions = []
    print(indre_df.columns)
    current_input = initial_window['Water Level'].to_numpy()
    last_know_date = indre_df['Timestamp'].loc[indre_df.index[-1]] # Finder den aller sidste dato i indre_df 

    for _ in range(steps_ahead):
        current_window = current_input[-window_size:].reshape(1, window_size, 1)
        predicted_value = model.predict(current_window)
            
        current_input = np.roll(current_input, shift=-1)
        current_input[-1] = predicted_value  # Insert the prediction into the last position

        incremented_date = pd.to_datetime(last_know_date, format='%d-%m-%Y %H:%M') + pd.Timedelta(minutes=30)
        last_know_date = incremented_date
        predictions.append([incremented_date, predicted_value[0][0]])
        # print(incremented_date)

    return np.array(predictions)

# initial_window = indre_df[-10:]
# steps_ahead = 48
# 
# indre_future_predictions = predict_future(indre_model, initial_window, window_size=5, steps_ahead=steps_ahead, indre_df=indre_df)
# indre_future_predictions_df = pd.DataFrame(indre_future_predictions, columns=['Timestamp', 'Water Level'])
# indre_future_predictions_df['Timestamp'] = indre_future_predictions_df['Timestamp'].dt.strftime('%d-%m-%Y %H:%M')
# indre_future_predictions_df['Water Level'] = indre_future_predictions_df['Water Level'].round(2).astype(float)
# 
# ydre_future_predictions = predict_future(ydre_model, initial_window, window_size=5, steps_ahead=steps_ahead, indre_df=ydre_df)
# ydre_future_predictions_df = pd.DataFrame(ydre_future_predictions, columns=['Timestamp', 'Water Level'])
# ydre_future_predictions_df['Timestamp'] = ydre_future_predictions_df['Timestamp'].dt.strftime('%d-%m-%Y %H:%M')
# ydre_future_predictions_df['Water Level'] = ydre_future_predictions_df['Water Level'].round(2).astype(float)
# 
# print(indre_future_predictions_df)
# print(ydre_future_predictions_df)
# 
# predictions(indre_future_predictions_df, ydre_future_predictions_df)"""