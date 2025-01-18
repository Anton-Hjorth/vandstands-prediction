import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Conv1D, Flatten, GRU, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error as mse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from convert_csv import indre_vandstande, ydre_vandstande

# sliceing ser sådan her ud [start:stop:step] vi bruger ikke stop
indre_df = indre_vandstande[0::6] # data for hvert 30. minut 


indre_df.index = pd.to_datetime(indre_df['Timestamp'], format='%d-%m-%Y %H:%M')
water_level = indre_df['Water Level']

# df = fx indre_df eller ydre_df
# window_size = 
def df_to_X_y(df, window_size):
    df_as_np = df.to_numpy()
    X = []
    y = []

    for i in range(len(df_as_np) - window_size):
        row = [[a] for a in df_as_np[i:i + window_size]]
        X.append(row)
        label = df_as_np[i + window_size]
        y.append(label)
    return np.array(X), np.array(y)

window_size = 5
indre_X, indre_y = df_to_X_y(water_level, window_size)
# print(indre_X.shape, indre_y.shape)

indre_X_train, indre_y_train = indre_X[:60000], indre_y[:60000]
indre_X_val, indre_y_val = indre_X[60000:70000], indre_y[60000:70000]
indre_X_test, indre_y_test = indre_X[100000:], indre_y[100000:]


indre_model = Sequential()
indre_model.add(InputLayer((5, 1)))
indre_model.add(LSTM(128, return_sequences=True))
indre_model.add(Dropout(0.85))
indre_model.add(LSTM(64))
indre_model.add(Dropout(0.4))
indre_model.add(Dense(16, activation='tanh'))
indre_model.add(Dense(8, activation='tanh'))
indre_model.add(Dense(1, activation='linear'))
# relu = avoids negative values
# sigmoid = values between 0 and 1
# tanh = values between -1 and 1
# linear = values between -inf and inf

# indre_model_CP = ModelCheckpoint('Models/indre_model.keras', save_best_only=True)
# indre_model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])
# indre_model.fit(indre_X_train, indre_y_train, validation_data=(indre_X_val, indre_y_val), epochs=10, callbacks=[indre_model_CP])

indre_model = load_model('Models/indre_model.keras')

def plot_predictions(predictions):
    plt.figure(figsize=(14, 7))
    plt.plot(predictions['Test Predictions'][:100], label='Val Predictions' )
    plt.plot(predictions['Actuals'][:100], label='Actuals')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Test Predictions vs Actuals')
    plt.legend()
    plt.show()


# indre_predictions = indre_model.predict(indre_X_test).flatten()
# future_prediction_results = pd.DataFrame(data={'Test Predictions': indre_predictions, 'Actuals': indre_y_test})
# plot_predictions(future_prediction_results)

def predictions(predictions_df):
    # Set the 'Timestamp' column as the index
    predictions_df.set_index('Timestamp', inplace=True)
    
    # Plot the DataFrame
    plot = predictions_df.plot(title="Future Predictions")
    plt.xlabel('Timestamp')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.show()


def predict_future(model, initial_window, window_size, steps_ahead, indre_df):
    predictions = []

    current_input = initial_window['Water Level'].to_numpy()
    last_know_date = indre_df['Timestamp'].loc[indre_df.index[-1]] # Finder den aller sidste dato i indre_df 

    for _ in range(steps_ahead):
        current_window = current_input[-window_size:].reshape(1, window_size, 1)
        predicted_value = model.predict(current_window)
            
        current_input = np.roll(current_input, shift=1)
        current_input[-1] = predicted_value  # Insert the prediction into the last position

        incremented_date = pd.to_datetime(last_know_date, format='%d-%m-%Y %H:%M') + pd.Timedelta(minutes=30)
        last_know_date = incremented_date
        predictions.append([incremented_date, predicted_value[0][0]])
        # print(incremented_date)

    return np.array(predictions)

# initial_window = indre_df[-10:]
# steps_ahead = 48
# 
# future_predictions = predict_future(indre_model, initial_window, window_size=5, steps_ahead=steps_ahead)
# future_predictions_df = pd.DataFrame(future_predictions, columns=['Timestamp', 'Water Level'])
# future_predictions_df['Timestamp'] = future_predictions_df['Timestamp'].dt.strftime('%d-%m-%Y %H:%M')
# future_predictions_df['Water Level'] = future_predictions_df['Water Level'].round(2).astype(float)
# 
# predictions(future_predictions_df)
# print(future_predictions_df)
# predictions(indre_predictions)


def multi_step(step_size, batch_size, model, window_size, initial_window, indre_df, learning_rate, epochs):
    def train_model(model, indre_X_train, indre_y_train, indre_X_val, indre_y_val, learning_rate, epochs):
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])
        model.fit(indre_X_train, indre_y_train, validation_data=(indre_X_val, indre_y_val), epochs=epochs)

    def predict_future(model, initial_window, window_size, steps_ahead, indre_df):
        predictions = []

        current_input = initial_window['Water Level'].to_numpy()
        last_know_date = indre_df['Timestamp'].loc[indre_df.index[-1]]  # Find last timestamp

        for _ in range(steps_ahead):
            current_window = current_input[-window_size:].reshape(1, window_size, 1)
            predicted_value = model.predict(current_window)
                
            current_input = np.roll(current_input, shift=1)
            current_input[-1] = predicted_value  # Insert the prediction into the last position

            incremented_date = pd.to_datetime(last_know_date, format='%d-%m-%Y %H:%M') + pd.Timedelta(minutes=30)
            last_know_date = incremented_date
            predictions.append([incremented_date, predicted_value[0][0]])

        return np.array(predictions)

    water_level = indre_df['Water Level']
    for i in range(batch_size):
        indre_X, indre_y = df_to_X_y(water_level, window_size)
        indre_X_train, indre_y_train = indre_X[:60000], indre_y[:60000]
        indre_X_val, indre_y_val = indre_X[60000:70000], indre_y[60000:70000]

        train_model(indre_model, indre_X_train, indre_y_train, indre_X_val, indre_y_val, learning_rate, epochs)
        future_predictions = predict_future(model, initial_window, window_size, step_size, indre_df)
        future_predictions_df = pd.DataFrame(future_predictions, columns=['Timestamp', 'Water Level'])
        future_predictions_df['Timestamp'] = future_predictions_df['Timestamp'].dt.strftime('%d-%m-%Y %H:%M')
        future_predictions_df['Water Level'] = future_predictions_df['Water Level'].round(2).astype(float)

        print(indre_df.columns)
        print(future_predictions_df.columns)

        future_predictions_df = future_predictions_df[['Timestamp', 'Water Level']]
        indre_df = indre_df[['Timestamp', 'Water Level']]

        frames = [indre_df, future_predictions_df]
        indre_df = pd.concat(frames, ignore_index=True)
        print(f"Process {i}/{batch_size}")

    return indre_df

# ====== fix der er alt for mange curves og dataerne er værre end før kig længere oppe det der er commentet ud ======== # 
step_size = 4
batch_size = 8
model = indre_model
initial_window = indre_df[-10:]
learning_rate = 0.001
epochs = 1

df = multi_step(step_size, batch_size, model, window_size, initial_window, indre_df, learning_rate, epochs)
df = df.tail(batch_size*step_size)
print(df)
predictions(df)