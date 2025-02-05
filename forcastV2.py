import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Conv1D, Flatten, GRU, Dropout # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore
from tensorflow.keras.metrics import RootMeanSquaredError # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.layers import TimeDistributed # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from convert_csv import indre_vandstande, ydre_vandstande
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import mean_squared_error as mse

def plot_predictions1(model, x, y, start=0, end=100):
    predictions = model.predict(x).flatten()
    y = y.flatten()  # Flatten the actual values
    df = pd.DataFrame(data={'Predictions': predictions, 'Actuals': y})
    plt.plot(df['Predictions'][start:end])
    plt.plot(df['Actuals'][start:end])
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.savefig('PredictionsV2.png')
    #plt.show()
    return df, mse(y, predictions)

# sliceing ser sådan her ud [start:stop:step] vi bruger ikke stop
indre_df = indre_vandstande[0::6] # data for hvert 30. minut 
ydre_df = ydre_vandstande[0::3] # data for hvert 30. minut

indre_df.index = pd.to_datetime(indre_df['Timestamp'], format='%d-%m-%Y %H:%M')
ydre_df.index = pd.to_datetime(ydre_df['Timestamp'], format='%d-%m-%Y %H:%M')

indre_water_level = indre_df['Water Level']
ydre_water_level = ydre_df['Water Level']

indre_df = pd.DataFrame({'Water Level': indre_water_level})
indre_df['Seconds'] = indre_df.index.map(pd.Timestamp.timestamp)

ydre_df = pd.DataFrame({'Water Level': ydre_water_level})
ydre_df['Seconds'] = ydre_df.index.map(pd.Timestamp.timestamp)

indre_df['Timestamp'] = indre_df.index # enssure that timestamp is part of the data frame
ydre_df['Timestamp'] = ydre_df.index


day = 60*60*24 # seconds
year = 365.2425*day # seconds/year

indre_df['Day sin'] = np.sin(indre_df['Water Level'] * (2 * np.pi / day))
indre_df['Day cos'] = np.cos(indre_df['Water Level'] * (2 * np.pi / day))
indre_df['Year sin'] = np.sin(indre_df['Water Level'] * (2 * np.pi / year))
indre_df['Year cos'] = np.cos(indre_df['Water Level'] * (2 * np.pi / year))

ydre_df['Day sin'] = np.sin(ydre_df['Water Level'] * (2* np.pi / day))
ydre_df['Day cos'] = np.cos(ydre_df['Water Level'] * (2 * np.pi / day))
ydre_df['Year sin'] = np.sin(ydre_df['Water Level'] * (2 * np.pi / year))
ydre_df['Year cos'] = np.cos(ydre_df['Water Level'] * (2 * np.pi / year))

indre_df = indre_df.drop('Seconds', axis=1)
ydre_df = ydre_df.drop('Seconds', axis=1)

def df_to_X_y(df, window_size, future_steps):
    df_as_np = df.to_numpy()
    X, y = [], []

    for i in range(len(df_as_np) - window_size - future_steps):
        X.append(df_as_np[i:i + window_size])
        y.append(df_as_np[i + window_size:i + window_size + future_steps])

    return np.array(X), np.array(y)

window_size = 1  # Number of previous time steps to consider
future_steps = 48  # Number of future time steps to predict

indre_X, indre_y = df_to_X_y(indre_water_level, window_size, future_steps)
ydre_X, ydre_y = df_to_X_y(ydre_water_level, window_size, future_steps)

indre_X_train, indre_y_train = indre_X[:60000], indre_y[:60000]
indre_X_val, indre_y_val = indre_X[60000:70000], indre_y[60000:70000]
indre_X_test, indre_y_test = indre_X[100000:], indre_y[100000:]

ydre_X_train, ydre_y_train = ydre_X[:60000], ydre_y[:60000]
ydre_X_val, ydre_y_val = ydre_X[60000:70000], ydre_y[60000:70000]
ydre_X_test, ydre_y_test = ydre_X[100000:], ydre_y[100000:]

indre_model = Sequential([
    InputLayer((window_size, 1)),  # Input shape (7 time steps, 1 feature)
    LSTM(64, return_sequences=True),
    LSTM(32, return_sequences=False),
    Dense(128, activation='relu'),
    Dense(future_steps, activation='linear')  # Output 48 future steps
])

ydre_model = Sequential([
    InputLayer((window_size, 1)),
    LSTM(64, return_sequences=True),
    LSTM(32, return_sequences=False),
    Dense(128, activation='relu'),
    Dense(future_steps, activation='linear')
])

print(indre_model.summary())
print(ydre_model.summary())

indre_model_CP = ModelCheckpoint('Models/indre_modelV2.keras', save_best_only=True)
indre_model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])
indre_model.fit(indre_X_train, indre_y_train, validation_data=(indre_X_val, indre_y_val), epochs=1, callbacks=[indre_model_CP])

ydre_model_CP = ModelCheckpoint('Models/ydre_modelV2.keras', save_best_only=True)
ydre_model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])
ydre_model.fit(ydre_X_train, ydre_y_train, validation_data=(ydre_X_val, ydre_y_val), epochs=1, callbacks=[ydre_model_CP])

indre_model = load_model('Models/indre_modelV2.keras')
ydre_model = load_model('Models/ydre_modelV2.keras')

indre_future_predictions = indre_model.predict(indre_X_test[:1])  # Predict next 48 values
ydre_future_predictions = ydre_model.predict(ydre_X_test[:1])  # Predict next 48 values

# Add 0.5 to each value of the indre future predictions
indre_future_predictions += 0.5

print(indre_future_predictions)
print(ydre_future_predictions)

# Plot the predictions
plt.figure(figsize=(14, 7))

# Plot Indre predictions
plt.plot(indre_future_predictions.flatten(), label='Indre Future Predictions', color='blue')

# Plot Ydre predictions
plt.plot(ydre_future_predictions.flatten(), label='Ydre Future Predictions', color='red')

# Set labels and title
plt.xlabel('Time Steps')
plt.ylabel('Predicted Values')
plt.title('Indre vs Ydre Future Predictions')
plt.legend()

# Show the plot
plt.tight_layout()
plt.savefig('PredictionsV2.png')
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