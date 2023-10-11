#! /usr/bin/env python

"""Running the simulation in notebook causes the notebook to crash
(Kernel restarting) due to memory exhausion.

Due to the above issue we test the manual implementation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pmdarima
import statsmodels.tsa.stattools as st
import statsmodels.api as sm
import itertools
import queue
import threading
import multiprocessing
import sys
from time import time, sleep
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.diagnostic import acorr_ljungbox

resultQueue = multiprocessing.Queue()
processes = list()

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

def test_model(data, building_id,  params, seasonal_params, result_queue):
    """ 
    Test a single model instance and put the results in a Queue.
    
    A queue provides in built synchronization
    """
    model = sm.tsa.statespace.SARIMAX(data, order=params, seasonal_order=seasonal_params, 
                                      enforce_stationarity=False, enforce_invertibility=False, trend=None)
    res = model.fit()
    pred = res.get_prediction(start=start_date)
    pred_values = pred.predicted_mean
    mae = mean_absolute_error(data[start_date:], pred_values)
    mse = mean_squared_error(data[start_date:], pred_values)
    rmse = np.sqrt(mse)
    result_queue.put([building_id, params, seasonal_params, res.aic, res.bic, mae, mse, rmse])

def seasonal_arima_simulation_threaded(data, building_id, start_date='2022-01-01 00:00:00', **kwargs):
    """
    Perform simulation on a multicore environment in a manner that 
    bypasses Python's  Global Interpreter Lock (GIL)
    
    Significant performance gains observered 4X reduction in time (61.45mins to 14.23mins) 
    Tested by simulating 3 buildings in single and multi-processor code
    """
    from warnings import filterwarnings
    filterwarnings("ignore")

    # Generate combinations of various (p, d, q) parameters to simulate
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    pdq_seasonal = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    #print(f'simulating a total of {len(pdq) * len(pdq_seasonal) * len(masks.keys())} models')
    start = time()
    for orderARIMA in pdq:
        for seasonalARIMA in pdq_seasonal:
            try:
                # spawn processes to make use of multi-processor context
                p = multiprocessing.Process(target=test_model, args=(data, building_id, orderARIMA, seasonalARIMA, resultQueue))
                processes.append(p)
                p.start()
            except:
                print("An error occurred: ", sys.exc_info()[0])
    for process in processes:
        process.join()
    print(f'took {time() - start} to complete simulating building {building_id}')
    df = pd.DataFrame([resultQueue.get() for i in range(resultQueue.qsize())], columns=['building_id', 'ARMA Order',
                                                                                        'Seasonal Order', 'AIC', 'BIC', 'MAE', 'MSE', 'RMSE'])
    df.sort_values('AIC', inplace=True)
    return df


# Run simulations
data = data_preprocessing()
print(type(data), data.keys())

simulation_res_df = pd.DataFrame(columns=data.keys())
start = time()
for i, k in enumerate(data.keys()):
    print(f'Simulating model for Building {k}')
    sim_res = seasonal_arima_simulation_threaded(data[k]['Consumption'], k, start_date='2022-01-01 00:00:00')
    print(f'recording results')
    simulation_res_df[k] = sim_res.iloc[0] # For each building, store the best model
print(f'>>> simulation for all buildings took {(time()- start)} secs')
print(f'Storing results')
simulation_res_df.to_csv('./simulation_results.csv')
