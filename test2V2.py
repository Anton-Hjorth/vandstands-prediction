import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# Generate sine/cosine data for testing
def generate_sine_cosine_data(length=100000, frequency=0.01):
    timestamps = np.arange(length)
    sine_data = np.sin(2 * np.pi * frequency * timestamps)
    cosine_data = np.cos(2 * np.pi * frequency * timestamps)
    return sine_data, cosine_data

# Generate sine and cosine data
sine_data, cosine_data = generate_sine_cosine_data()

# Convert to DataFrame for consistency with your original code
sine_df = pd.DataFrame(sine_data, columns=['Value'])
cosine_df = pd.DataFrame(cosine_data, columns=['Value'])

# Add a timestamp column for plotting
sine_df['Timestamp'] = pd.date_range(start='2023-01-01', periods=len(sine_df), freq='T')
cosine_df['Timestamp'] = pd.date_range(start='2023-01-01', periods=len(cosine_df), freq='T')

# Function to convert DataFrame to X and y for LSTM
def df_to_X_y(df, window_size):
    df_as_np = df['Value'].to_numpy()
    X = []
    y = []

    for i in range(len(df_as_np) - window_size):
        row = [[a] for a in df_as_np[i:i + window_size]]
        X.append(row)
        label = df_as_np[i + window_size]
        y.append(label)
    return np.array(X), np.array(y)

# Define window size
window_size = 10

# Prepare data for sine and cosine
sine_X, sine_y = df_to_X_y(sine_df, window_size)
cosine_X, cosine_y = df_to_X_y(cosine_df, window_size)

# Split into train/val/test sets
train_size = int(0.7 * len(sine_X))
val_size = int(0.2 * len(sine_X))

sine_X_train, sine_y_train = sine_X[:train_size], sine_y[:train_size]
sine_X_val, sine_y_val = sine_X[train_size:train_size + val_size], sine_y[train_size:train_size + val_size]
sine_X_test, sine_y_test = sine_X[train_size + val_size:], sine_y[train_size + val_size:]

cosine_X_train, cosine_y_train = cosine_X[:train_size], cosine_y[:train_size]
cosine_X_val, cosine_y_val = cosine_X[train_size:train_size + val_size], cosine_y[train_size:train_size + val_size]
cosine_X_test, cosine_y_test = cosine_X[train_size + val_size:], cosine_y[train_size + val_size:]

# Define LSTM model
def build_model():
    model = Sequential()
    model.add(InputLayer((window_size, 1)))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(8, activation='tanh'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])
    return model

# Build and train sine model
sine_model = build_model()
sine_model_CP = ModelCheckpoint('Models/sine_model.keras', save_best_only=True)
sine_model.fit(sine_X_train, sine_y_train, validation_data=(sine_X_val, sine_y_val), epochs=1, batch_size=32, callbacks=[sine_model_CP])

# Build and train cosine model
cosine_model = build_model()
cosine_model_CP = ModelCheckpoint('Models/cosine_model.keras', save_best_only=True)
cosine_model.fit(cosine_X_train, cosine_y_train, validation_data=(cosine_X_val, cosine_y_val), epochs=1, batch_size=32, callbacks=[cosine_model_CP])

# Load best models
sine_model = load_model('Models/sine_model.keras')
cosine_model = load_model('Models/cosine_model.keras')

# Make predictions
sine_predictions = sine_model.predict(sine_X_test)
cosine_predictions = cosine_model.predict(cosine_X_test)

# Plot predictions
def plot_predictions(sine_predictions, cosine_predictions, sine_y_test, cosine_y_test, max_xticks=10):
    plt.figure(figsize=(14, 7))
    
    # Plot sine predictions
    plt.plot(sine_y_test[:100], label='Sine Actuals', color='blue')
    plt.plot(sine_predictions[:100], label='Sine Predictions', color='orange', linestyle='--')
    
    # Plot cosine predictions
    plt.plot(cosine_y_test[:100], label='Cosine Actuals', color='green')
    plt.plot(cosine_predictions[:100], label='Cosine Predictions', color='red', linestyle='--')
    
    # Set labels and title
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title('Sine and Cosine Predictions vs Actuals')
    plt.legend()
    
    # Limit the number of x-axis labels
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=max_xticks))
    
    # Show the plot
    plt.tight_layout()
    plt.savefig('Sine_Cosine_Predictions.png')
    plt.show()

# Call the plotting function
plot_predictions(sine_predictions, cosine_predictions, sine_y_test, cosine_y_test)