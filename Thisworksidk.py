

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

    # Create a new column 'Price_BE_next_day' and shift 'Price_BE' by 1
    data['Price_BE_next_day'] = data['Price_BE'].shift(-1)
    data.dropna(inplace=True)
    # Drop the last row
    data.drop(data.tail(1).index, inplace=True)

    n_hours = 24
    start = datetime.datetime(2021, 1, 1, 0, 0)
    end = datetime.datetime(2023, 1, 25, 23, 0)
    X = data['Price_BE'][start:end].resample('1H').mean().values.reshape(-1, n_hours)


    start = datetime.datetime(2021, 1, 1, 0, 0)
    end = datetime.datetime(2023, 1, 25, 23, 0)
    y = data['Price_BE_next_day'][start:end].resample('1H').mean().values.reshape(-1, n_hours)
    # Split the data into features and target
    #X = data[['Load_FR', 'Gen_FR', 'Price_CH','Wind_BE', 'Solar_BE', 'Load_BE', 'Price_BE']]
    #X = data['Price_BE']
    #y = data['Price_BE_next_day']

    # Split the data into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create the model 
    model = Sequential()
    #model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    #model.add(Dense(32, activation='relu'))
    #model.add(Dense(1, activation='linear'))
    model.add(Dense(32, input_shape=(n_hours,1), activation='relu'))
    model.add(Dense(1, activation='linear'))

    rprop = RMSprop(learning_rate=0.01, rho=0.9, epsilon=1e-07)
    model.compile(loss='mean_squared_error', optimizer=rprop)

    # Fit the model
    output_training = model.fit(X, y, epochs=100, batch_size=10, verbose=0)
    mse = output_training.history['loss'][-1]
    print('- mse is %.4f' % mse + ' @ ' + str(len(output_training.history['loss'])))

    predict_nn = model.predict(X)
   

    # Plot the results
    plt.figure()
    plt.plot(y.flatten(), color='blue', label='actual price')
    plt.plot(predict_nn.flatten(), color='red', label='forecast NN')
    plt.legend(frameon=False)
    plt.show()