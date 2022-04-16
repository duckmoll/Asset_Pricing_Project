#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:05:11 2022

@author: mattharding
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import statistics
from statistics import mean
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import numpy


analytical = pd.read_feather("cleaned_df_2015.feather")

## -------------------------------------
# Begin by testing parameters on validation set 
#   Using rsme to determine best number of components to use
# -------------------------------------


# def run_pcr_valid(df,yr,var_ex):
#     training = df[pd.to_datetime(df['date'], format='%Y-%m-%d').dt.to_period('Y')  <= str(yr)]
#     training_y = training["exret_1m"]
#     training_x = training.drop( ['exret_1m','index','permno','date'],axis = 1)
#     training_x = StandardScaler().fit_transform(training_x)
    
#     valid = df[pd.to_datetime(df['date'], format='%Y-%m-%d').dt.to_period('Y')  == str(yr+1)]
#     valid_y = valid["exret_1m"]
#     valid_x = valid.drop( ['exret_1m','index','permno','date'],axis = 1)
#     valid_x = StandardScaler().fit_transform(valid_x)
    
    
#     regr_pca = PCA(var_ex)
#     regr_pca.fit(training_x)
#     training_x = regr_pca.transform(training_x)
#     valid_x = regr_pca.transform(valid_x)
    
#     regr = LinearRegression()
#     regr.fit(training_x, training_y)
#     pred = pd.DataFrame(regr.predict(valid_x))
        

#     new_df = pd.concat([valid_y,pred],axis = 1)
#     new_df.columns = ['actual', 'pred']
#     ret = {"df" : new_df,
#            "comp":regr_pca.components_}

#     return(ret)

# comps = [.25,.3,.4,.5,.6,.7,.8]
# rmse_vec = []
# out_df = {"pct_var" : comps,
#           "num_comp" : [0,0,0,0,0,0,0],
#           "rmse" : [0,0,0,0,0,0,0]}
# out_df = pd.DataFrame(out_df)
# count = 0  
# for c in comps:
#     pcr = run_pcr_valid(analytical,2018,c)
#     df = pcr["df"]
#     num_comp = pcr["comp"]
#     num_comp = num_comp.shape[0]
#     rmse = sqrt((df["actual"] - df["pred"]).mean()**2)
#     out_df.at[count,:] = [c,num_comp,rmse]
#     count += 1
comps = [11,28,56,56]



# Function for running pcr
def run_pca(df,yr,comps):
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
    
    # Run PCR
    regr_pca = PCA(comps)
    regr_pca.fit(training_x)
    training_x = regr_pca.transform(training_x)
    testing_x = regr_pca.transform(testing_x)
    
    regr = LinearRegression()
    regr.fit(training_x, training_y)
    pred = pd.DataFrame(regr.predict(testing_x))
    
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
pca_2015 = run_pca(analytical,2015,comps[0])
pca_2015.to_csv("pca_res/pca_2015.csv")

pca_2016 = run_pca(analytical,2016,comps[1])
pca_2016.to_csv("pca_res/pca_2016.csv")

pca_2017 = run_pca(analytical,2017,comps[2])
pca_2017.to_csv("pca_res/pca_2017.csv")

pca_2018 = run_pca(analytical,2018,comps[3])
pca_2018.to_csv("pca_res/pca_2018.csv")

# #pls_2019 = run_pls(analytical,2019)
# #pls_2019.to_csv("pls_res/pls_2019.csv")




