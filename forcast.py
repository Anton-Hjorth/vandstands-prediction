
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Conv1D, Flatten, GRU, Dropout # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore
from tensorflow.keras.metrics import RootMeanSquaredError # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from convert_csv import indre_vandstande, ydre_vandstande
from matplotlib.ticker import MaxNLocator

# sliceing ser sådan her ud [start:stop:step] vi bruger ikke stop
indre_df = indre_vandstande[0::6] # data for hvert 30. minut 
ydre_df = ydre_vandstande[0::3] # data for hvert 30. minut

indre_df.index = pd.to_datetime(indre_df['Timestamp'], format='%d-%m-%Y %H:%M')
ydre_df.index = pd.to_datetime(ydre_df['Timestamp'], format='%d-%m-%Y %H:%M')

indre_water_level = indre_df['Water Level']
ydre_water_level = ydre_df['Water Level']

# day = 60*60*24 # seconds
# year = 365.2425*day # seconds/year
# 
# indre_df['Day sin'] = np.sin(indre_df['Seconds'] * (2* np.pi / day))
# indre_df['Day cos'] = np.cos(indre_df['Seconds'] * (2 * np.pi / day))
# indre_df['Year sin'] = np.sin(indre_df['Seconds'] * (2 * np.pi / year))
# indre_df['Year cos'] = np.cos(indre_df['Seconds'] * (2 * np.pi / year))

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
indre_X, indre_y = df_to_X_y(indre_water_level, window_size)
ydre_X, ydre_y = df_to_X_y(ydre_water_level, window_size)
# print(indre_X.shape, indre_y.shape)

indre_X_train, indre_y_train = indre_X[:60000], indre_y[:60000]
indre_X_val, indre_y_val = indre_X[60000:70000], indre_y[60000:70000]
indre_X_test, indre_y_test = indre_X[100000:], indre_y[100000:]

ydre_X_train, ydre_y_train = ydre_X[:60000], ydre_y[:60000]
ydre_X_val, ydre_y_val = ydre_X[60000:70000], ydre_y[60000:70000]
ydre_X_test, ydre_y_test = ydre_X[100000:], ydre_y[100000:]

indre_model = Sequential()
indre_model.add(InputLayer((5, 1)))
indre_model.add(LSTM(128, return_sequences=True))
indre_model.add(Dropout(0.85))
indre_model.add(LSTM(64))
indre_model.add(Dropout(0.4))
indre_model.add(Dense(16, activation='tanh'))
indre_model.add(Dense(8, activation='tanh'))
indre_model.add(Dense(1, activation='linear'))

ydre_model = Sequential()
ydre_model.add(InputLayer((5, 1)))
ydre_model.add(LSTM(128, return_sequences=True))
ydre_model.add(Dropout(0.85))
ydre_model.add(LSTM(64))
ydre_model.add(Dropout(0.4))
ydre_model.add(Dense(16, activation='tanh'))
ydre_model.add(Dense(8, activation='tanh'))
ydre_model.add(Dense(1, activation='linear'))

# relu = avoids negative values
# sigmoid = values between 0 and 1
# tanh = values between -1 and 1
# linear = values between -inf and inf

# indre_model_CP = ModelCheckpoint('Models/indre_model.keras', save_best_only=True)
# indre_model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])
# indre_model.fit(indre_X_train, indre_y_train, validation_data=(indre_X_val, indre_y_val), epochs=10, callbacks=[indre_model_CP])
# 
# ydre_model_CP = ModelCheckpoint('Models/ydre_model.keras', save_best_only=True)
# ydre_model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])
# ydre_model.fit(ydre_X_train, ydre_y_train, validation_data=(ydre_X_val, ydre_y_val), epochs=10, callbacks=[ydre_model_CP])

indre_model = load_model('Models/indre_model.keras')
ydre_model = load_model('Models/ydre_model.keras')

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
    plt.savefig('Predictions.png')
    plt.show()


def multi_step(step_size, batch_size, model, window_size, initial_window, df, learning_rate, epochs, frequency, amplitude, shift):
    def train_model(model, X_train, y_train, X_val, y_val, learning_rate, epochs):
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)
        return model

    def predict_future(model, initial_window, window_size, steps_ahead, df, frequency):
        predictions = []

        current_input = np.copy(initial_window['Water Level'].to_numpy())

        if len(current_input) < window_size:
            padding = np.full((window_size - len(current_input),), current_input[-1])
            current_input = np.concatenate((padding, current_input))

        last_known_date = pd.to_datetime(df['Timestamp'].iloc[-1], format='%d-%m-%Y %H:%M') 

        for i in range(steps_ahead):
            current_window = current_input[-window_size:].reshape(1, window_size, 1)
            predicted_value = model.predict(current_window).flatten()[0]
            
            sin_component = np.sin(2 * np.pi * (i / steps_ahead) * frequency) * 0.4
            predicted_value = predicted_value * 0.7 + sin_component * 0.3

            current_input = np.roll(current_input, shift=1)
            current_input[-1] = predicted_value  # Insert the prediction into the last position

            last_known_date += pd.Timedelta(minutes=30)
            predictions.append([last_known_date, predicted_value])

        return np.array(predictions)
    
     # amplitude = height frequency = speed/oscillation

    df = df.copy()  # Ensure it's a new DataFrame
    df.loc[:, 'Sin_Feature'] = np.sin(np.linspace(0, 2 * np.pi * frequency, len(df))) + shift
    df.loc[:, 'Water Level'] += amplitude * df['Sin_Feature']

    water_level = df['Water Level']
    
    for i in range(batch_size):
        X, y = df_to_X_y(water_level, window_size)
        X_train, y_train = X[:60000+i], y[:60000+i]
        X_val, y_val = X[40000+i:70000+i], y[40000+i:70000+i]

        train_model(model, X_train, y_train, X_val, y_val, learning_rate, epochs)
        future_predictions = predict_future(model, initial_window, window_size, step_size, df, frequency)

        future_predictions_df = pd.DataFrame(future_predictions, columns=['Timestamp', 'Water Level'])
        future_predictions_df['Timestamp'] = future_predictions_df['Timestamp'].dt.strftime('%d-%m-%Y %H:%M')

        future_predictions_df['Water Level'] = future_predictions_df['Water Level'].round(2).astype(float)

        future_predictions_df = future_predictions_df[['Timestamp', 'Water Level']]
        df = df[['Timestamp', 'Water Level']]

        # print("future predictions", future_predictions_df.head())

        df = pd.concat([df[['Timestamp', 'Water Level']], future_predictions_df[['Timestamp', 'Water Level']]], ignore_index=True)
        initial_window = df[-10**i:]
        print(f"Process {i}/{batch_size}")

    print("Final df:\n", df.head())
    print("Final df columns:", df.columns)
    print("Final df types:\n", df.dtypes)
    return df.reset_index(drop=True)

step_size = 48 # 1
batch_size = 1 # 48
indre_initial_window = indre_df[-10:]
ydre_initial_window = ydre_df[-10:]
learning_rate = 0.001
epochs = 1
indre_displacement = 0.5
ydre_displacement = 0.5

indre_frequency, indre_amplitude, indre_shift = 1.5, 0.4, 0.5
ydre_frequency, ydre_amplitude, ydre_shift = 1.9, 2, 0.25
# increase amplitude = increase height
# decrease amplitude = decrease height
# increase frequency = increase speed "tighter waves"
# decrease frequency = decrease speed "wider waves"

indre_df = multi_step(step_size, batch_size, indre_model, window_size, indre_initial_window, indre_df, learning_rate, epochs, indre_frequency, indre_amplitude, indre_shift)
indre_df = indre_df.tail(batch_size*step_size)
indre_df = indre_df.reset_index(level=[0]) 
indre_array = indre_df[['Timestamp', 'Water Level']].values
indre_df = indre_df.drop(columns=['index']) 

ydre_df = multi_step(step_size, batch_size, ydre_model, window_size, ydre_initial_window, ydre_df, learning_rate, epochs, ydre_frequency, ydre_amplitude, ydre_shift)
ydre_df = ydre_df.tail(batch_size*step_size)
ydre_df = ydre_df.reset_index(level=[0]) 
ydre_array = ydre_df[['Timestamp', 'Water Level']].values
ydre_df = ydre_df.drop(columns=['index']) 

predictions(indre_df, ydre_df)