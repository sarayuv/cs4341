#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
'''Write your code here '''
df = pd.read_csv("")

'''end of student code'''
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
total_df = df[['Year', 'GDP', 'Adult Mortality', 'Alcohol', ' BMI ', 'Schooling', 'Life expectancy ', 'Status', 'Country']]
total_df.isna().any()
total_df = total_df.dropna()

for status in ["Developing", "Developed"]:
    x_dim = 1
    #Step 1:You should define the train_x, each row of it represents a year of the 5 features of a country with 5 columns. The Year column should be used to select the samples.
    #Step 2:Define train_y, each row of it represents a year of Life expectancy of a country. The Year column should be used to select the samples.
    #Step 3:Define a LinearRegression model, and fit it using train_X and train_y.
    #Step 4:Calculate rmse and r2_score using the fitted model.
    #Step 5:Print the coefficients of the linear regression model
    '''Write your code here '''
    
    status_df = total_df[total_df['Status'] == status]
    
    train_x_gdp = np.array([status_df[status_df.Year <= 2013].GDP**i 
                            for i in range(1, x_dim + 1)]).transpose(1, 0).reshape((-1, x_dim))
    train_x_adult_mortality = np.array([status_df[status_df.Year <= 2013]['Adult Mortality']**i 
                                        for i in range(1, x_dim + 1)]).transpose(1, 0).reshape((-1, x_dim))
    train_x_alcohol = np.array([status_df[status_df.Year <= 2013].Alcohol**i 
                                for i in range(1, x_dim + 1)]).transpose(1, 0).reshape((-1, x_dim))
    train_x_bmi = np.array([status_df[status_df.Year <= 2013][' BMI ']**i 
                            for i in range(1, x_dim + 1)]).transpose(1, 0).reshape((-1, x_dim))
    train_x_schooling = np.array([status_df[status_df.Year <= 2013].Schooling**i 
                                  for i in range(1, x_dim + 1)]).transpose(1, 0).reshape((-1, x_dim))
    
    train_x = np.hstack((train_x_gdp, train_x_adult_mortality, train_x_alcohol, train_x_bmi, train_x_schooling))
    train_y = np.array(status_df[(status_df.Year <= 2013)]['Life expectancy '])

    model = LinearRegression()
    model.fit(train_x, train_y)
    
    train_rmse = mean_squared_error(model.predict(train_x), train_y)
    train_r2_score = model.score(train_x, train_y)
    
    '''end of student code'''
    print(f'Status = {status}, Training data, RMSE={train_rmse:.3f}, R2={train_r2_score:.3f}')
    for feature_i, feature in enumerate(['GDP', 'Adult Mortality', 'Alcohol', ' BMI ', 'Schooling']):
        print(f'coef for {feature} = {model.coef_[feature_i]:.7f}')
    #Step 1: Define test_x and test_y by selecting the remaining years of the data
    #Step 2: Use model.predict to generate the prediction
    #Step 3: Calculate rmse and r2_score on test_x and test_y.
    '''Write your code here '''

    test_x_gdp = np.array([status_df[status_df.Year > 2013].GDP**i 
                            for i in range(1, x_dim + 1)]).transpose(1, 0).reshape((-1, x_dim))
    test_x_adult_mortality = np.array([status_df[status_df.Year > 2013]['Adult Mortality']**i 
                                        for i in range(1, x_dim + 1)]).transpose(1, 0).reshape((-1, x_dim))
    test_x_alcohol = np.array([status_df[status_df.Year > 2013].Alcohol**i 
                                for i in range(1, x_dim + 1)]).transpose(1, 0).reshape((-1, x_dim))
    test_x_bmi = np.array([status_df[status_df.Year > 2013][' BMI ']**i 
                            for i in range(1, x_dim + 1)]).transpose(1, 0).reshape((-1, x_dim))
    test_x_schooling = np.array([status_df[status_df.Year > 2013].Schooling**i 
                                  for i in range(1, x_dim + 1)]).transpose(1, 0).reshape((-1, x_dim))
    
    test_x = np.hstack((test_x_gdp, test_x_adult_mortality, test_x_alcohol, test_x_bmi, test_x_schooling))
    test_y = np.array(status_df[(status_df.Year > 2013)]['Life expectancy '])

    test_predict = model.predict(test_x)
    
    test_rmse = mean_squared_error(test_predict, test_y)**0.5
    test_r2_score = model.score(test_x, test_y)
    
    '''end of student code'''
    print(f'Status = {status}, Testing data, RMSE={test_rmse:.3f}, R2={test_r2_score:.3f}')
