#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.cross_decomposition import PLSRegression
import pandas as pd
from math import sqrt
import statistics
from statistics import mean
from sklearn.preprocessing import StandardScaler
import numpy as np

# Dataset
analytical = pd.read_feather("cleaned_df_2015.feather")


# -------------------------------------
# Begin by testing parameters on validation set 
#   Using rsme to determine best number of components to use
# -------------------------------------


# def run_pls_valid(df,yr,comps):
#     training = df[pd.to_datetime(df['date'], format='%Y-%m-%d').dt.to_period('Y')  <= str(yr)]
#     training_y = training["exret_1m"]
#     training_x = training.drop( ['exret_1m','index','permno','date'],axis = 1)
#     training_x = StandardScaler().fit_transform(training_x)
    
#     valid = df[pd.to_datetime(df['date'], format='%Y-%m-%d').dt.to_period('Y')  == str(yr+1)]
#     valid_y = valid["exret_1m"]
#     valid_x = valid.drop( ['exret_1m','index','permno','date'],axis = 1)
#     valid_x = StandardScaler().fit_transform(valid_x)
    
#     regr = PLSRegression(comps)
#     regr.fit(training_x, training_y)
#     pred = pd.DataFrame(regr.predict(valid_x))
#     pred.reset_index(drop=True, inplace=True)
#     valid_y.reset_index(drop=True, inplace=True)
#     new_df = pd.concat([valid_y,pred],axis = 1)
#     new_df.columns = ['actual', 'pred']

#     return(new_df)
# comps = [2,3,4,5,10,15,20,25,30]
# rmse_vec = []

# for c in comps:
#     pls = run_pls_valid(analytical,2018,c)
#     rmse = sqrt((pls["actual"] - pls["pred"]).mean()**2)
#     rmse_vec.append(rmse)

# List with best number of components for eac year
comps = [2,2,4,2]

# Function for running pls
def run_pls(df,yr,comps):
    # Create df of training data
    training = df[pd.to_datetime(df['date'], format='%Y-%m-%d').dt.to_period('Y')  <= str(yr)]
    training_y = training["exret_1m"]
    training_x = training.drop( ['exret_1m','index','permno','date'],axis = 1)
    training_x = StandardScaler().fit_transform(training_x)
    
    # Create df of testing daa
    testing = df[pd.to_datetime(df['date'], format='%Y-%m-%d').dt.to_period('Y')  == str(yr+2)]
    testing_y = testing[["exret_1m","permno"]]
    testing_x = testing.drop( ['exret_1m','index','permno','date'],axis = 1)
    testing_x = StandardScaler().fit_transform(testing_x)
    
    # Run PLS
    regr = PLSRegression(comps)
    regr.fit(training_x, training_y)
    
    # Create predictions
    pred = pd.DataFrame(regr.predict(testing_x))
    
    # Reset indexes to merge 
    pred.reset_index(drop=True, inplace=True)
    testing_y.reset_index(drop=True, inplace=True)
    
    # Merge testing & predictions & return
    new_df = pd.concat([testing_y,pred],axis = 1)
    new_df.columns = ['actual','permno', 'pred']

    return(new_df)


# Run function for each year of training
pls_2015 = run_pls(analytical,2015,comps[0])
pls_2015.to_csv("pls_res/pls_2015.csv")

pls_2016 = run_pls(analytical,2016,comps[1])
pls_2016.to_csv("pls_res/pls_2016.csv")

pls_2017 = run_pls(analytical,2017,comps[2])
pls_2017.to_csv("pls_res/pls_2017.csv")

pls_2018 = run_pls(analytical,2018,comps[3])
pls_2018.to_csv("pls_res/pls_2018.csv")


