import pandas as pd
import numpy as np
import datetime
import time

from tda import auth, client
import json
from config import api_key, redirect_uri, token_path, account_id, executable_path

# Machine learning libraries
from sklearn.svm import SVC
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix, recall_score, precision_score, classification_report
import talib as ta

import seaborn as sns
sns.set(style="darkgrid")
import matplotlib.pyplot as plt


def data_cleaning_engineering(df):
    # Drop the rows with zero volume traded
    df = df.drop(df[df['volume'] == 0].index)
    # Create a variable n with a value of 10
    n = 10
    # Create a column by name, RSI and assign the calculation of RSI to it
    df['RSI'] = ta.RSI(np.array(df['close'].shift(1)), timeperiod=n)
    
    # Create a column by name, SMA and assign the SMA calculation to it
    df['SMA'] = df['close'].shift(1).rolling(window=n).mean()
    # Create a column by name, Corr and assign the calculation of correlation to it
    df['Corr'] = df['close'].shift(1).rolling(window=n).corr(df['SMA'].shift(1))
    # Create a column by name, SAR and assign the SAR calculation to it
    df['SAR'] = ta.SAR(np.array(df['high'].shift(1)), np.array(df['low'].shift(1)), 0.2, 0.2)
    # Create a column by name, ADX and assign the ADX calculation to it
    df['ADX'] = ta.ADX(np.array(df['high'].shift(1)), np.array(df['low'].shift(1)),
                       np.array(df['open']), timeperiod=n)
    
    # Create columns high, low and close with previous minute's OHLC data
    df['Prev_High'] = df['high'].shift(1)
    df['Prev_Low'] = df['low'].shift(1)
    df['Prev_Close'] = df['close'].shift(1)

    # Create columns 'OO' with the difference between the current minute's open and last minute's open
    df['OO'] = df['open']-df['open'].shift(1)
    # Create columns 'OC' with the difference between the current minute's open and last minute's close
    df['OC'] = df['open']-df['Prev_Close']
    
    # (close-low)/(high-low)
    df['Normalized'] = (df['close']-df['low'])/(df['high']-df['low'])
    
    # calculate slope of regression for Close price of last 3,5,10,20 min
    for day,reg_period in zip([3,5,10,20], ['reg_3','reg_5', 'reg_10', 'reg_20']):
        df[reg_period] =  df['close'].rolling(window=day).apply(get_slope, raw=False)
        
    # Create a column 'Ret' with the calculation of returns
    df['Ret'] = (df['open'].shift(-1)-df['open'])/df['open']

    # Create n columns and assign
    for i in range(1, n):
        df['return%i' % i] = df['Ret'].shift(i)
    df.iloc[-1:] = df.iloc[-1:].fillna(0)
    
    # Additional Cleaning 
    # Change the value of 'Corr' to -1 if it is less than -1
    df.loc[df['Corr'] < -1, 'Corr'] = -1
    # Change the value of 'Corr' to 1 if it is greater than 1
    df.loc[df['Corr'] > 1, 'Corr'] = 1
    # Drop the NaN values
    df = df.dropna()
    
    # Create a column by name, 'Signal' and initialize with 0
    df['Signal'] = 0
    # Assign a value of 1 to 'Signal' column for the quantile with the highest returns
    df.loc[df['Ret'] > df['Ret'].quantile(q=0.66), 'Signal'] = 1
    # Assign a value of -1 to 'Signal' column for the quantile with the lowest returns
    df.loc[df['Ret'] < df['Ret'].quantile(q=0.34), 'Signal'] = -1
    
    return df

def train_classifier_predict(df):
    # Use drop method to drop the columns
    
    X = df.drop(['close', 'Signal', 'high', 'volume', 'low', 'Ret'], axis=1)
    # Create a variable which contains all the 'Signal' values
    y = df['Signal']
    
    # Create a variable split that stores 80% of the length of the dataframe
    t = .98
    split = int(t*len(df))
    
    # Test variables for 'c' and 'g'
    #Setting the different values to test within C, Gamma and Kernel
    c = [10, 100, 1000, 10000]
    g = [1e-2, 1e-1, 1e0]
    # Intialise the parameters
    parameters = {'svc__C': c,
                  'svc__gamma': g,
                  'svc__kernel': ['rbf']}
    #Creating the step by step pipeline
    # Create the 'steps' variable with the pipeline functions
    steps = [('scaler', StandardScaler()), ('svc', SVC())]
    # Pass the 'steps' to the Pipeline function
    pipeline = Pipeline(steps)
    #Creating a randomized function to help to find the best parameters.
    # Call the RandomizedSearchCV function and pass the parameters
    rcv = RandomizedSearchCV(pipeline, parameters, cv=TimeSeriesSplit(n_splits=10))
    
    # Call the 'fit' method of rcv and pass the train data to it
    rcv.fit(X.iloc[:split], y.iloc[:split])
    # Call the 'best_params_' method to obtain the best parameters of C
    best_C = rcv.best_params_['svc__C']
    # Call the 'best_params_' method to obtain the best parameters of kernel
    best_kernel = rcv.best_params_['svc__kernel']
    # Call the 'best_params_' method to obtain the best parameters of gamma
    best_gamma = rcv.best_params_['svc__gamma']
    
    # Create a new SVC classifier
    cls = SVC(C=best_C, kernel = best_kernel, gamma=best_gamma)
    # Instantiate the StandardScaler
    ss1 = StandardScaler()
    # Pass the scaled train data to the SVC classifier
    cls.fit(ss1.fit_transform(X.iloc[:split]), y.iloc[:split])
    
    prediction = cls.predict(ss1.transform(X.iloc[-1:]))
    timestamp = X.iloc[-1:].index
    return timestamp, prediction


def get_slope(array):
    y = np.array(array)
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    return slope

def get_account_acess(api_key, redirect_uri, token_path, account_id, executable_path):
    try:
        c = auth.client_from_token_file(token_path, api_key)
    except FileNotFoundError:
        from selenium import webdriver
        with webdriver.Chrome(executable_path = executable_path) as driver:
            c = auth.client_from_login_flow(
                driver, api_key, redirect_uri, token_path)
    return c




def correct_signals(series):
    original_signal = series.to_list()
    corrected_signal = []

    for record_index, current_record in enumerate(original_signal):
        if record_index == 0:
            corrected_signal.append(current_record)
            current_s = current_record
            continue

        previous_record = corrected_signal[record_index - 1]

        record_to_add = current_record
        if previous_record == current_record or current_s==current_record:
            record_to_add = 0

        if record_to_add != current_s and record_to_add != 0:
            current_s = record_to_add

        corrected_signal.append(record_to_add)
    return corrected_signal


def plot_predict_signal(df_pred, col_orig_signal, col_pred_signal, figsize=(14,4)):
    
    fig, ax = plt.subplots(figsize=figsize)
    plot = sns.lineplot(x='datetime', y='open', data=df_pred, marker='o', label='Actual Data')
    plot = sns.lineplot(x='datetime', y='Open_shift', data=df_pred, marker='o', label='Predictions')
    # Annotate label points 
    for x,y,m in df_pred[['datetime','open', col_orig_signal]].values:
            ax.text(x,y+.4,m, size=7)
    for x,y,m in df_pred[['datetime','Open_shift', col_pred_signal]].values:
            ax.text(x,y-.7,m, size=7)
    plt.setp(plot.get_xticklabels(), rotation=90, size =8)
    plt.setp(plot.get_yticklabels(), size =8)
    plt.title('Actual Signal and Predicted Signal for TESLA')
    plt.ylabel('Stock Open Price ($)')
    ax.legend()
    plt.show()