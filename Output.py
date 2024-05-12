# Plot the relevant dates and output price values as csv file with no other information

# Relevant dates:
# 23/01/2024, at 00:00, to 25/01/2024, at 23:00 (hourly data)

# Output file: 'Price_BE_23_01_2024_25_01_2024.csv'

#imports

import pandas as pd

def Output(data):
    
    # Get the relevant dates
    start_date = '23/01/2024 00:00'
    end_date = '25/01/2024 23:00'
    
    # Filter the data for the relevant dates
    data_filtered = data.loc[start_date:end_date]

    # Output the data to a csv file
    data_filtered.to_csv('Price_BE_23_01_2024_25_01_2024.csv', index=False)

    # Check that the number of rows is 72
    print("Number of rows in the output file:", data_filtered.shape[0])

    if data_filtered.shape[0] == 72:
        print("Output file created successfully.")
    else:
        print("Error creating output file.")
    
