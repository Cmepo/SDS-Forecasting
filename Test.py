# Import csv values from Predictions.csv and print them

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Read data from Predictions.csv
# no headers
df = pd.read_csv('Predictions.csv', header=None)

print(df)
