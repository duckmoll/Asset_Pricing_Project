#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 18:33:51 2022

@author: mattharding
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Datasets
analytical = pd.read_feather("cleaned_df_2015.feather")
#clust = pd.read_feather("cleaned_df_2015_kmeans.feather")

def run_lin_mod(df,yr):
    # Split data for training
    training = df[pd.to_datetime(df['date'], format='%Y-%m-%d').dt.to_period('Y')  <= str(yr)]
    # Get target variable
    training_y = training["exret_1m"]
    # Standarize variables
    training_x = training.drop( ['exret_1m','index','permno','date'],axis = 1)
    training_x = StandardScaler().fit_transform(training_x)
    # Get testing data
    testing = df[pd.to_datetime(df['date'], format='%Y-%m-%d').dt.to_period('Y') == str(yr + 1)]
    # Get target for testing
    testing_y = testing[["exret_1m",'permno']]
    # Standardize testing variables
    testing_x = testing.drop(['exret_1m','index','permno','date'],axis = 1)
    testing_x = StandardScaler().fit_transform(testing_x)
    # Initiate linear regressions
    regr = LinearRegression()
    # Fit to training data
    regr.fit(training_x, training_y)
    # Make predictions on testing data
    pred = pd.DataFrame(regr.predict(testing_x))
    # Save actual and prediction data for r^2 calculations
    # Index needs to be reset to match
    pred.reset_index(drop=True, inplace=True)
    testing_y.reset_index(drop=True, inplace=True)
    
    new_df = pd.concat([testing_y,pred],axis = 1)
    new_df.columns = ['actual','permno', 'pred']

    return(new_df)

# run function on different subset of data
# Original data
lin_2015 = run_lin_mod(analytical,2015)
lin_2015.to_csv("lin_reg_res/lin_2015.csv")

lin_2016 = run_lin_mod(analytical,2016)
lin_2016.to_csv("lin_reg_res/lin_2016.csv")

lin_2017 = run_lin_mod(analytical,2017)
lin_2017.to_csv("lin_reg_res/lin_2017.csv")

lin_2018 = run_lin_mod(analytical,2018)
lin_2018.to_csv("lin_reg_res/lin_2018.csv")

lin_2019 = run_lin_mod(analytical,2019)
lin_2019.to_csv("lin_reg_res/lin_2019.csv")

# Kmeans cluster data
# km_2015 = run_lin_mod(clust,2015)
# km_2015.to_csv("kmeans_lin_reg/km_2015.csv")

# km_2016 = run_lin_mod(clust,2016)
# km_2016.to_csv("kmeans_lin_reg/km_2016.csv")

# km_2017 = run_lin_mod(clust,2017)
# km_2017.to_csv("kmeans_lin_reg/km_2017.csv")

# km_2018 = run_lin_mod(clust,2018)
# km_2018.to_csv("kmeans_lin_reg/km_2018.csv")

# km_2019 = run_lin_mod(clust,2019)
# km_2019.to_csv("kmeans_lin_reg/km_2019.csv")

