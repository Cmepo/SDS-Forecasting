# Need three data sets
# 1. Training data
# 2. Validation data
# 3. Test data

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def Forecast(data):

    # Split the data into training, validation, and test sets
    train, test = train_test_split(data, test_size=0.2, shuffle=False)
    train, val = train_test_split(train, test_size=0.25, shuffle=False)

    # Extract the features and target variable
    X_train = train.drop(columns=['Price_BE'])
    y_train = train['Price_BE']

    X_val = val.drop(columns=['Price_BE'])
    y_val = val['Price_BE']

    X_test = test.drop(columns=['Price_BE'])
    y_test = test['Price_BE']

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_val)

    # Calculate the mean squared error
    mse = mean_squared_error(y_val, y_pred)
    print(f"\nMean squared error: {mse}")

    # Plot the predictions
    plt.figure(figsize=(10, 6))
    plt.plot(val.index, y_val, label='Actual')
    plt.plot(val.index, y_pred, label='Predicted', linestyle='--')
    plt.title('Price_BE Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price_BE')
    plt.legend()
    plt.show()

    # Make predictions on the test set
    y_pred_test = model.predict(X_test)

    # Calculate the mean squared error on the test set
    # Eliminate NaN values
    y_test = y_test[~np.isnan(y_test)]
    y_pred_test = y_pred_test[~np.isnan(y_pred_test)]
    # Make sure both arrays have the same length
    min_len = min(len(y_test), len(y_pred_test))
    y_test = y_test[:min_len]
    y_pred_test = y_pred_test[:min_len]
    mse_test = mean_squared_error(y_test, y_pred_test)
    print(f"\nMean squared error on test set: {mse_test}")

    # Plot the predictions on the test set
    plt.figure(figsize=(10, 6))
    plt.plot(test.index[:min_len], y_test, label='Actual')
    plt.plot(test.index[:min_len], y_pred_test, label='Predicted', linestyle='--')
    plt.title('Price_BE Prediction on Test Set')
    plt.xlabel('Date')
    plt.ylabel('Price_BE')
    plt.legend()

    plt.show()




    
