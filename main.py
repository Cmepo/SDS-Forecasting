# Description: Main file to run the project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Modules
import DataValidation as dv
import LinearForecast as lf
import NeuralNetwork as nn
import Output as op
import Neural as nn2

# Load the data
df1 = pd.read_csv('Dataset For Forecasting Assignment.csv')

dv.DataValidation(df1)
#lf.LinearForecast(df1)
#prediction = nn.NeuralNetwork(df1)

# Output data to csv file 
#op.Output(prediction)

prediction = nn2.Neural(df1)
