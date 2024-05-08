# Neural network for predicting day ahead prices

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop, SGD


def NeuralNetwork(data):

    
    # Predict the next day, based on available load for import and export

    data.reset_index(inplace=True)
    # Get index again from csv date
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data.set_index('Date', inplace=True)
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)

    # Create a new column 'Price_BE_next_day' and shift 'Price_BE' by 1
    data['Price_BE_next_day'] = data['Price_BE'].shift(-1)
    data.dropna(inplace=True)
    # Drop the last row
    data.drop(data.tail(1).index, inplace=True)

    
    # Split the data into features and target
    X = data[['Load_FR', 'Gen_FR', 'Price_CH','Wind_BE', 'Solar_BE', 'Load_BE', 'Price_BE']]
    #X = data['Price_BE']
    y = data['Price_BE_next_day']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create the model 
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))


    rprop = RMSprop(learning_rate=0.01, rho=0.9, epsilon=1e-07)
    model.compile(loss='mean_squared_error', optimizer=rprop)

    # Fit the model
    output_training = model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=0)
    mse = output_training.history['loss'][-1]
    print('- mse is %.4f' % mse + ' @ ' + str(len(output_training.history['loss'])))

    predict_nn = model.predict(X_test)
    test_index = range(len(y_train), len(y_train) + len(y_test))
    
    # Plot results for testing data
    plt.figure(figsize=(10, 5))
    plt.plot(test_index, y_test, color='blue', label='Actual Price (Testing)')
    plt.plot(test_index, predict_nn, color='red', label='Predicted Price (Testing)')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Testing Data: Actual vs Predicted Price Over Time')
    plt.legend()
    plt.show()