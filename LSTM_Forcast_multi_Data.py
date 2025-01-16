import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Conv1D, Flatten, GRU
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

df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
temp = df['T (degC)']

def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []

    for i in range(len(df_as_np) - window_size):
        row = [[a] for a in df_as_np[i:i + 5]]
        X.append(row)
        label = df_as_np[i + 5]
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

val_predictions = model1.predict(X_val).flatten()
val_results = pd.DataFrame(data={'Val Predictions': val_predictions, 'Actuals': y_val})
#print(val_results)

test_predictions = model1.predict(X_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions': test_predictions, 'Actuals': y_test})
# print(test_results)

def plot_predictions1(model, x, y, start=0, end=100):
    predictions = model.predict(x).flatten()
    df = pd.DataFrame(data={'Predictions': predictions, 'Actuals': y})
    plt.plot(df['Predictions'][start:end])
    plt.plot(df['Actuals'][start:end])
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()
    return df, mse(y, predictions)

# plot_predictions1(model1, X_test, y_test)

model2 = Sequential()
model2.add(InputLayer((5, 1)))
model2.add(Conv1D(64, kernel_size=2))
model2.add(Flatten())
model2.add(Dense(8, 'relu'))
model2.add(Dense(1, 'linear'))

# print(model2.summary())

# =============== comment this block to avoid training the model vv=============
# cp2 = ModelCheckpoint('model2.keras', save_best_only=True)
# model2.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
# model2.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp2])
# =============== comment this block to avoid training the model ^^=============


model3 = Sequential()
model3.add(InputLayer((5, 1)))
model3.add(GRU(64))
model3.add(Dense(8, 'relu'))
model3.add(Dense(1, 'linear'))

# print(model3.summary()) # fere params betyder at det tager længere tid at træne "tror jeg"

# =============== comment this block to avoid training the model vv=============
# cp3 = ModelCheckpoint('model3.keras', save_best_only=True)
# model3.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
# model3.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp3])
# =============== comment this block to avoid training the model ^^=============

temp_df = pd.DataFrame({'Temperature': temp})
temp_df['Seconds'] = temp_df.index.map(pd.Timestamp.timestamp)

day = 60*60*24 # seconds
year = 365.2425*day # seconds/year

temp_df['Day sin'] = np.sin(temp_df['Seconds'] * (2* np.pi / day))
temp_df['Day cos'] = np.cos(temp_df['Seconds'] * (2 * np.pi / day))
temp_df['Year sin'] = np.sin(temp_df['Seconds'] * (2 * np.pi / year))
temp_df['Year cos'] = np.cos(temp_df['Seconds'] * (2 * np.pi / year))

temp_df = temp_df.drop('Seconds', axis=1)

def df_to_X_y2(df, window_size=6):
    df_as_np = df.to_numpy()
    X = []
    y = []

    for i in range(len(df_as_np) - window_size):
        row = [r for r in df_as_np[i:i + window_size]]
        X.append(row)
        label = df_as_np[i + window_size][0]
        y.append(label)
    return np.array(X), np.array(y)

X2, y2 = df_to_X_y2(temp_df)
# print(X2.shape, y2.shape)

X2_train, y2_train = X2[:60000], y2[:60000]
X2_val, y2_val = X2[60000:65000], y2[60000:65000]
X2_test, y2_test = X2[65000:], y2[65000:]
# print(X2_train.shape, y2_train.shape, X2_val.shape, y2_val.shape, X2_test.shape, y2_test.shape)

temp_training_mean = np.mean(X2_train[:, :, 0])
temp_training_std = np.std(X2_train[:, :, 0])

def preprocess_X(X):
    X[:, :, 0] = (X[:, :, 0] - temp_training_mean) / temp_training_std
    return X

# print(preprocess_X(X2_train))
# print(preprocess_X(X2_val))
# print(preprocess_X(X2_test))

model4 = Sequential()
model4.add(InputLayer((6, 5)))
model4.add(LSTM(64))
model4.add(Dense(8, 'relu'))
model4.add(Dense(1, 'linear'))

# print(model4.summary())

# =================== comment this block to avoid training the model vv===================
# cp4 = ModelCheckpoint('model4.keras', save_best_only=True)
# model4.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
# model4.fit(X2_train, y2_train, validation_data=(X2_val, y2_val), epochs=10, callbacks=[cp4])
# =================== comment this block to avoid training the model ^^===================

# plot_predictions1(model4, X2_test, y2_test)

p_temp_df = pd.concat([df['p (mbar)'], temp_df], axis=1)

def df_to_X_y3(df, window_size=7):
    df_as_np = df.to_numpy()
    X = []
    y = []

    for i in range(len(df_as_np) - window_size):
        row = [r for r in df_as_np[i:i + window_size]]
        X.append(row)
        label = [df_as_np[i + window_size][0], df_as_np[i + window_size][1]]
        y.append(label)
    return np.array(X), np.array(y)

X3, y3 = df_to_X_y3(p_temp_df)
# print(X3.shape, y3.shape)

X3_train, y3_train = X3[:60000], y3[:60000]
X3_val, y3_val = X3[60000:65000], y3[60000:65000]
X3_test, y3_test = X3[65000:], y3[65000:]
# print(X3_train.shape, y3_train.shape, X3_val.shape, y3_val.shape, X3_test.shape, y3_test.shape)

p_training_mean3 = np.mean(X3_train[:, :, 0])
p_training_std3 = np.std(X3_train[:, :, 0])

temp_training_mean3 = np.mean(X3_train[:, :, 1])
temp_training_std3 = np.std(X3_train[:, :, 1])

def preprocess3(X):
    X[:, :, 0] = (X[:, :, 0] - p_training_mean3) / p_training_std3
    X[:, :, 1] = (X[:, :, 1] - temp_training_mean3) / temp_training_std3


def preprocess_output3(y):
    y[:, 0] = (y[:, 0] - p_training_mean3) / p_training_std3
    y[:, 1] = (y[:, 1] - temp_training_mean3) / temp_training_std3
    return y

preprocess3(X3_train)
preprocess3(X3_val)
preprocess3(X3_test)

preprocess_output3(y3_train)
preprocess_output3(y3_val)
preprocess_output3(y3_test)

model5 = Sequential()
model5.add(InputLayer((7, 6)))
model5.add(LSTM(64))
model5.add(Dense(8, 'relu'))
model5.add(Dense(2, 'linear'))

# print(model5.summary())

# == comment this block to avoid training the model vv==
# cp5 = ModelCheckpoint('model5.keras', save_best_only=True)
# model5.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
# model5.fit(X3_train, y3_train, validation_data=(X3_val, y3_val), epochs=10, callbacks=[cp5])
# == comment this block to avoid training the model ^^==

def plot_predictions2(model, X, y, start=0, end=100):
    predictions = model.predict(X)
    p_preds, temp_preds = predictions[:, 0], predictions[:, 1]
    p_actuals, temp_actuals = y[:, 0], y[:, 1]

    df = pd.DataFrame(data={'Temperature Predictions': temp_preds,
                            'Temperature Actuals':temp_actuals,
                            'Pressure Predictions': p_preds,
                            'Pressure Actuals': p_actuals
                            })
    
    plt.plot(df['Temperature Predictions'][start:end])
    plt.plot(df['Temperature Actuals'][start:end])
    plt.plot(df['Pressure Predictions'][start:end])
    plt.plot(df['Pressure Actuals'][start:end])
    # plt.show()
    return df[start:end]

plot_predictions2(model5, X3_test, y3_test)

def postprocess_temp(arr):
    arr = (arr * temp_training_std3) + temp_training_mean3
    return arr

def postprocess_p(arr):
    arr = (arr * p_training_std3) + p_training_mean3
    return arr

def plot_predictions2(model, X, y, start=0, end=100):
    predictions = model.predict(X)
    p_preds, temp_preds = postprocess_p(predictions[:, 0]), postprocess_temp(predictions[:, 1])
    p_actuals, temp_actuals = postprocess_p(y[:, 0]), postprocess_temp(y[:, 1])

    df = pd.DataFrame(data={'Temperature Predictions': temp_preds,
                            'Temperature Actuals':temp_actuals,
                            'Pressure Predictions': p_preds,
                            'Pressure Actuals': p_actuals
                            })
    plt.plot(df['Temperature Predictions'][start:end])
    plt.plot(df['Temperature Actuals'][start:end])
    plt.plot(df['Pressure Predictions'][start:end])
    plt.plot(df['Pressure Actuals'][start:end])
    # plt.show()
    return df[start:end]

post_process_df = plot_predictions2(model5, X3_test, y3_test)
print(post_process_df)

model6 = Sequential()
model6.add(InputLayer((7, 6)))
model6.add(LSTM(32, return_sequences=True))
model6.add(LSTM(64))
model6.add(Dense(8, 'relu'))
model6.add(Dense(2, 'linear'))

# == comment this block to avoid training the model vv==
# cp6 = ModelCheckpoint('model6.keras', save_best_only=True)
# model6.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
# model6.fit(X3_train, y3_train, validation_data=(X3_val, y3_val), epochs=10, callbacks=[cp6])
# == comment this block to avoid training the model ^^==

model7 = Sequential()
model7.add(InputLayer((7, 6)))
model7.add(Conv1D(64, kernel_size=2, activation='relu'))
model7.add(Flatten())
model7.add(Dense(8, 'relu'))
model7.add(Dense(2, 'linear'))

# =================== comment this block to avoid training the model vv===================
cp7 = ModelCheckpoint('model7.keras', save_best_only=True)
model7.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
model7.fit(X3_train, y3_train, validation_data=(X3_val, y3_val), epochs=10, callbacks=[cp7])
# =================== comment this block to avoid training the model ^^===================