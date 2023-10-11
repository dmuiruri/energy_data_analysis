#! /usr/bin/env python

"""Running the simulation in notebook causes the notebook to crash
(Kernel restarting) due to memory exhausion.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pmdarima
import statsmodels.tsa.stattools as st
import statsmodels.api as sm
from time import time, sleep
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.diagnostic import acorr_ljungbox


def data_preprocessing():
    print('Data pre-processing')
    rdata = pd.read_csv('./Granlund_data_anon_040422_v2.csv', sep=';', decimal=',')
    data = rdata.set_index('Time')

    # Pick only relevant columns
    # Convert the index to a datetime object type and sort it
    data = data[['Consumption', 'Temp_outside', 'anon_id']]
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)

    # Create a maskk of the building IDs
    masks = dict()
    for unique_id in data['anon_id'].unique():
        masks[unique_id] = data['anon_id']==unique_id

    clean_data = {}
    for k in masks.keys():
        print(f'Processing data {k}')
        df = data[masks[k]][~data[masks[k]].index.duplicated()] # remove duplicates
        original_index = df.index # data[masks[k]][~data[masks[k]].index.duplicated()].index
        start_date = original_index[0]
        end_date = original_index[-1]
        new_index = pd.date_range(start=start_date, end=end_date, freq='H') # create a new reference index
        new_df = pd.DataFrame(index=new_index)   
        new_df['Consumption'] = df['Consumption']
        new_df['Temp_outside'] = df['Temp_outside']
        new_df.interpolate(method='slinear', inplace=True) # Handle NaNs with interpolation
        clean_data[k] = new_df
    return clean_data

def simulation(data):
    """Some hyperparameter tuning of m is required, 

    Documentation:The m parameter relates to the number of
    observations per seasonal cycle, and is one that must be known
    apriori. Typically, m will correspond to some recurrent
    periodicity such as: 7(daily), 12(monthly), 52(weekly)
    https://alkaline-ml.com/pmdarima/tips_and_tricks.html#period
    """
    empirical_model = pmdarima.auto_arima(data,
                                          start_p=1, start_q=1,
                                          max_p=7, max_q=7, d=1, m=7,
                                          start_P=0, seasonal=True,
                                          D=1, trace=True,
                                          error_action='ignore',  
                                          suppress_warnings=True, 
                                          stepwise=True)
    return empirical_model

def simulation_energy_temp(y, x):
    """
    Y: Energy
    X: Temperature
    Some hyperparameter tuning of m is required, 

    Documentation:The m parameter relates to the number of
    observations per seasonal cycle, and is one that must be known
    apriori. Typically, m will correspond to some recurrent
    periodicity such as: 7(daily), 12(monthly), 52(weekly)
    https://alkaline-ml.com/pmdarima/tips_and_tricks.html#period
    """
    empirical_model = pmdarima.auto_arima(y, x,
                                          start_p=1, start_q=1,
                                          max_p=7, max_q=7, d=1, m=5,
                                          start_P=0, seasonal=True,
                                          D=1, trace=True,
                                          error_action='ignore',  
                                          suppress_warnings=True, 
                                          stepwise=True)
    return empirical_model

if __name__ == '__main__':
    data = data_preprocessing()
    #arima_model = simulation(data[12]['Consumption'][:'2021'])
    energy_temp_model = simulation_energy_temp(data[12]['Consumption'][:'2021'],
                                               data[12]['Temp_outside'][:'2021'].values.reshape(-1, 1))
    
