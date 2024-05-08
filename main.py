# Description: Main file to run the project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Modules
import DataValidation as dv
import Forecasting as fc
import NeuralNetwork as nn

# Load the data
df1 = pd.read_csv('Dataset For Forecasting Assignment.csv')

dv.DataValidation(df1)
    #fc.Forecast(df1)

nn.NeuralNetwork(df1)



