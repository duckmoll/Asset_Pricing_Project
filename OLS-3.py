#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 15:31:21 2022

@author: mattharding
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


analytical = pd.read_feather("cleaned_df_2015.feather")

#df = analytical
#yr = 2015
def run_lin3_mod(df,yr):
    # Split data for training
    training = df[pd.to_datetime(df['date'], format='%Y-%m-%d').dt.to_period('Y')  <= str(yr)]
    # Get target variable
    training_y = training["exret_1m"]
    # Standarize variables
    training_x = training[['mom12m','bm','mvel1']]
    training_x = MinMaxScaler(feature_range = (-1,1)).fit_transform(training_x)
    # Get testing data
    testing = df[pd.to_datetime(df['date'], format='%Y-%m-%d').dt.to_period('Y') == str(yr + 1)]
    # Get target for testing
    testing_y = testing[["exret_1m","permno"]]
    # Standardize testing variables
    testing_x = testing[['mom12m','bm','mvel1']]
    testing_x = MinMaxScaler(feature_range = (-1,1)).fit_transform(testing_x)
    # Initiate linear regressions
    regr = LinearRegression()
    # Fit to training data
    regr.fit(training_x, training_y)
    # Make predictions on testing data
    pred = pd.DataFrame(regr.predict(testing_x))
    # Save actual and prediction data for r^2 calculations
    pred.reset_index(drop=True, inplace=True)
    testing_y.reset_index(drop=True, inplace=True)
    new_df = pd.concat([testing_y,pred],axis = 1)
    new_df.columns = ['actual','permno', 'pred']


    return(new_df)

# run function on different subset of data
lin3_2015 = run_lin3_mod(analytical,2015)
lin3_2015.to_csv("lin3_reg_res/lin3_2015.csv")

lin3_2016 = run_lin3_mod(analytical,2016)
lin3_2016.to_csv("lin3_reg_res/lin3_2016.csv")

lin3_2017 = run_lin3_mod(analytical,2017)
lin3_2017.to_csv("lin3_reg_res/lin3_2017.csv")

lin3_2018 = run_lin3_mod(analytical,2018)
lin3_2018.to_csv("lin3_reg_res/lin3_2018.csv")

lin3_2019 = run_lin3_mod(analytical,2019)
lin3_2019.to_csv("lin3_reg_res/lin3_2019.csv")

