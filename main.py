# Description: Main file to run the project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Modules
import DataValidation as dv
import NeuralNetwork as nn
import Neural as nn2

# Load the data
df1 = pd.read_csv('Dataset For Forecasting Assignment.csv')

dv.DataValidation(df1
#prediction = nn.NeuralNetwork(df1)

prediction = nn2.Neural(df1)
