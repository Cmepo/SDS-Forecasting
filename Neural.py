import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime

import keras_tuner
import keras

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.losses import MeanSquaredError
from keras import layers

def Neural(data):

    data.reset_index(inplace=True)
    # Get index again from csv date
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data.set_index('Date', inplace=True)
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)

    data['Price_BE_lag'] = data.Price_BE.shift(168)

    # Create new dataframe copying the old one
    data2 = data.copy()
    # drop na rows
    data.dropna(inplace=True)

    # Define the number of lagged time steps
    num_lags = 7  # Use lagged prices from the past 7 days (adjust as needed)

    # Create lagged features for Price_BE
    # for lag in range(1, num_lags + 1):
    #     data[f'Price_BE_lag_{lag}'] = data['Price_BE'].shift(lag)
    #     data2[f'Price_BE_lag_{lag}'] = data2['Price_BE'].shift(lag)
    
    # # Define the features to include in the model
    # train_features = ['Price_BE_lag_1','Price_BE_lag_2','Price_BE_lag_3',
    #                 'Price_BE_lag_4','Price_BE_lag_5','Price_BE_lag_6',
    #                 'Price_BE_lag_7','Load_FR', 'Gen_FR', 'Price_CH', 'Wind_BE', 'Solar_BE', 'Load_BE']

    train_features = ['Load_FR', 'Gen_FR', 'Price_CH', 'Wind_BE', 'Solar_BE', 'Load_BE', 'Price_BE_lag']
    train_target = ['Price_BE']
    # Prepare features and target variable
    X = data[train_features].values
    y = data[train_target].values

    print(X)
    print(y)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # # Build the neural network model
    # model = Sequential([
    #     Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    #     Dense(32, activation='relu'),
    #     Dense(1, activation='linear')
    # ])
    model = Sequential([
        Dense(128, input_shape=X_train_scaled.shape[1:]), 
        LeakyReLU(alpha=0.5),  
        Dense(64),
        LeakyReLU(alpha=0.5),
        Dense(32),
        LeakyReLU(alpha=0.5),
        Dense(1, activation='linear')  
    ])
    model.compile(optimizer=RMSprop(learning_rate=3e-6), loss='mean_squared_error', metrics=["mean_absolute_error"])

    # Train the model
    output_training = model.fit(X_train_scaled, y_train, epochs=150, batch_size=16, validation_split=0.2, verbose=1)
    mse = output_training.history['loss'][-1]
    print('- mse is %.4f' % mse + ' @ ' + str(len(output_training.history['loss'])))

    # Evaluate the model
    loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f'Test Mean Absolute Error: {mae}')

    # Make predictions
    predictions = model.predict(X_test_scaled)

    # Plot the predictions
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='True')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.show()

    # Make predictions on the following interval of time
    # Relevant dates:
    # 23/01/2024, at 00:00, to 25/01/2024, at 23:00 (hourly data)

    start_date = '2024-01-23 00:00'
    end_date = '2024-01-25 23:00'

    # Filter the data for the relevant dates
    data_filtered = data2.loc[start_date:end_date]

    # Prepare the features for prediction
    X_pred = data_filtered[train_features].values
    X_pred_scaled = scaler.transform(X_pred)

    # Make predictions
    predictions = model.predict(X_pred_scaled)
    #prediction_data['Price_BE_predicted'] = predictions

    # Plot the predictions
    plt.figure(figsize=(12, 6))
    plt.plot(data_filtered.index, predictions, label='Predicted')
    plt.show()

    predictions = pd.DataFrame(predictions)
    predictions.round(2)
    predictions.to_csv('Predictions.csv', index=False, header=False)

    # Print all values of the prediction
    # Set display options to show all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print(predictions)

    # Comparison dataframe from Comparison.csv
    comparison = pd.read_csv('Comparison.csv')
    print(comparison.columns)
    # Get the day ahead prices from the column "Day Ahead Auction (BE)"
    day_ahead_prices = comparison['Day Ahead Auction (BE)']
    print(day_ahead_prices)

    # plot the comparison
    plt.figure(figsize=(12, 6))
    plt.plot(predictions, label='Predicted')
    plt.plot(day_ahead_prices, label='True')
    plt.legend()
    plt.show()

    # Calculate the mean squared error between day ahead prices and predicted prices
    mse = np.mean((day_ahead_prices - predictions.squeeze())**2)
    print('Mean Squared Error:', mse)

