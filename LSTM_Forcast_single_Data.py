import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set a custom directory inside your repo where you have full access
repo_data_dir = os.path.join("D:", "CSV-Data", "tf-guid-data")  # Adjusted base directory

# Ensure the directory exists
os.makedirs(repo_data_dir, exist_ok=True)

# Download and extract the dataset into the custom folder
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True,
    cache_dir=repo_data_dir  # Save inside your custom directory
)

# Find the correct CSV file inside the extracted folder
extracted_folder = os.path.join(repo_data_dir, "datasets", "jena_climate_2009_2016_extracted")  # Correct path
csv_path = os.path.join(extracted_folder, "jena_climate_2009_2016.csv")  # CSV file path

# Read the CSV file
df = pd.read_csv(csv_path)
df = df[5::6]  # Subsample every 6th row

print(df.head())

df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
temp = df['T (degC)']

def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []

    for i in range(len(df_as_np) - window_size):
        row = [[a] for a in df_as_np[i:i + window_size]]
        X.append(row)
        label = df_as_np[i + window_size]
        y.append(label)
    return np.array(X), np.array(y)

WINDOW_SIZE = 5
X, y = df_to_X_y(temp, window_size=WINDOW_SIZE)
# print(X.shape, y.shape)

X_train, y_train = X[:60000], y[:60000]
X_val, y_val = X[60000:65000], y[60000:65000]
X_test, y_test = X[65000:], y[65000:]
# print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

model1 = Sequential()
model1.add(InputLayer((5, 1)))
model1.add(LSTM(64))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))
# print(model1.summary())

# =============== comment this block to avoid training the model vv=============
# cp = ModelCheckpoint('model1.keras', save_best_only=True)
# model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
# model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp])
# =============== comment this block to avoid training the model ^^=============

model1 = load_model('model1.keras')

train_predictions = model1.predict(X_train).flatten()
train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': y_train})
# print(train_results.head())

plt.figure(figsize=(14, 7))
plt.plot(train_results['Train Predictions'][50:100], label='Train Predictions' )
plt.plot(train_results['Actuals'][50:100], label='Actuals')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Train Predictions vs Actuals')
plt.legend()
plt.show()

val_predictions = model1.predict(X_val).flatten()
val_results = pd.DataFrame(data={'Val Predictions': val_predictions, 'Actuals': y_val})
#print(val_results)

# plt.figure(figsize=(14, 7))
# plt.plot(val_results['Val Predictions'][:100], label='Val Predictions' )
# plt.plot(val_results['Actuals'][:100], label='Actuals')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('Val Predictions vs Actuals')
# plt.legend()
# plt.show()

test_predictions = model1.predict(X_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions': test_predictions, 'Actuals': y_test})
# print(test_results)

# plt.figure(figsize=(14, 7))
# plt.plot(test_results['Test Predictions'][:100], label='Val Predictions' )
# plt.plot(test_results['Actuals'][:100], label='Actuals')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('Test Predictions vs Actuals')
# plt.legend()
# plt.show()

