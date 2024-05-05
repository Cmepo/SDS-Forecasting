# Module: DataValidation
# Check the data types, special values, data integrity, and headers of the dataset.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def DataValidation(data):


    ## Validate the data
    # Check data types
    print("\nData types:")
    print(data.dtypes)

    # Check for special values (e.g., NaN, infinity)
    print("\nSpecial values:")
    print(data.isnull().sum())  # Count NaN values
    print(data.isin([float('inf'), float('-inf')]).sum())  # Count infinity values

    # Verify data integrity
    print("\nData integrity:")
    print("Number of missing values:", data.isnull().sum().sum())  # Total count of missing values
    print("Number of duplicate rows:", data.duplicated().sum())  # Count duplicate rows

    # Convert 'Price_BE' column to numeric type, coercing non-numeric values to NaN
    data['Price_BE'] = pd.to_numeric(data['Price_BE'], errors='coerce')

    print("\nData types after conversion:")
    print(data.dtypes)

    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data.set_index('Date', inplace=True)

    # Make a list with all the headers
    print("\nHeaders:")
    headers = list(data.columns.values)
    print(headers)

    ## Plot
    # Plot 'Price_BE' against time, format the y-axis appropriately
    fig, ax = plt.subplots()

    ax.set_title('Price_BE against Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price_BE')

    ax.plot(data.index, data['Price_BE'], color='blue')

    # Formatting the y-axis
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))  # Set maximum number of ticks
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))  # Format the tick labels

    # Beautify the x-labels
    plt.gcf().autofmt_xdate()

    ax.plot(data.index, data['Price_BE'], color='blue')
    plt.show()

    # Print maximum and minimum price
    print('\nPrice_BE:')
    print('Maximum price:', data['Price_BE'].max())
    print('Minimum price:', data['Price_BE'].min())