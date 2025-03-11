# Neural network for predicting day ahead prices

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


def NeuralNetwork(data):

    
    # Predict the next day, based on available load for import and export

    data.reset_index(inplace=True)
    # Get index again from csv date
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data.set_index('Date', inplace=True)
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)


    # List of features to include in the model
    features = ['Price_BE', 'Price_CH', 'Load_FR', 'Gen_FR', 'Wind_BE', 'Solar_BE', 'Load_BE']

    X = data.drop(columns=features[0]).values  # Features excluding the target variable
    y = data['Price_BE'].values  # Target variable (price for the following day)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features (optional but recommended)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define custom loss function to handle missing values
    def custom_loss(y_true, y_pred):
        mask = ~tf.math.is_nan(y_true)  # Create a mask to exclude NaN values
        masked_true = tf.boolean_mask(y_true, mask)
        masked_pred = tf.boolean_mask(y_pred, mask)
        return MeanSquaredError()(masked_true, masked_pred)
    

    # Validation data
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)


    def build_model(hp):
        model = keras.Sequential()
        model.add(layers.Flatten(input_shape=(X_train_scaled.shape[1],)))
        # Tune the number of layers.
        for i in range(hp.Int("num_layers", 1, 3)):
            model.add(
                layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                    activation=hp.Choice("activation", ["leaky_relu","relu","tanh","sigmoid"]),
                    #input_shape=(X_train_scaled.shape[1],)
                )
            )
        model.compile(
            optimizer = RMSprop(learning_rate=1e-5, rho=0.9, epsilon=1e-07),
            loss="val_loss",
            metrics=["accuracy"],
        )
        return model
    
    build_model(keras_tuner.HyperParameters())

    tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_loss",
    max_trials=5,
    executions_per_trial=3,
    overwrite=True,
    directory="C:/Users/cmene/Documents/GitHub/SDS-Forecasting",
    project_name="KerasTuner")

    tuner.search_space_summary()
    tuner.search(X_train, y_train, epochs=3, validation_data = (X_val, y_val))

    # Get the top 2 models.
    models = tuner.get_best_models(num_models=2)
    best_model = models[0]
    best_model.summary()

    output_training = best_model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1, validation_data=(X_val, y_val))
    mse = output_training.history['loss'][-1]
    print('- mse is %.4f' % mse + ' @ ' + str(len(output_training.history['loss'])))

    predict_nn = best_model.predict(X_test)
    predict_nn = predict_nn[:,0]

    # Plot 
    test_index = range(len(y_train), len(y_train) + len(y_test))
    plt.figure(figsize=(10, 5))
    plt.plot(test_index, y_test, color='blue', label='Actual Price (Testing)')
    plt.plot(test_index, predict_nn, color='red', label='Predicted Price (Testing)')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Testing Data: Actual vs Predicted Price Over Time')
    plt.legend()
    plt.show()


    # Predict the prices for the following dates:
    # 23/01/2024, at 00:00, to 25/01/2024, at 23:00 (hourly data)

    start_date = '23/01/2024 00:00'
    end_date = '25/01/2024 23:00'

    # Filter the data for the relevant dates
    data_filtered = data.loc[start_date:end_date]

    print("\nSpecial values:")
    print(data_filtered.isnull().sum()) 
    print(data_filtered.isin([float('inf'), float('-inf')]).sum()) 

    data_filtered.drop('Price_BE_next_day', axis=1, inplace=True)
    data_filtered = data_filtered[['Price_BE','Load_FR', 'Gen_FR','Price_CH' ,'Wind_BE', 'Solar_BE', 'Load_BE']]

    print(data_filtered)
    print(X_test)

    # Predict the prices for the relevant dates
    predict_nn = best_model.predict(data_filtered)
    predict_nn = predict_nn[:,0]


    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(data_filtered.index, predict_nn, color='red', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Price Over Time')
    plt.legend()
    plt.show()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print(predict_nn)

    # Output the predicted values to csv file
    predict_nn.to_csv('Price_BE_23_01_2024_25_01_2024.csv', index=False)

    return predict_nn

