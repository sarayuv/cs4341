#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
'''Write your code here '''
df = pd.read_csv("")

'''end of student code'''
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
total_df = df[df.Country == 'Afghanistan'][['Year', 'GDP', 'Life expectancy ']]
for degree in [1,2,3,4]:
    '''Write your code here '''
    
    #Step 1:You should define the train_x, each row of it represents a year of GDP of Afghanistan,and each column of it represents a power of the GDP. The Year column should be used to select the samples.
    train_x = np.array([total_df[total_df.Year <= 2013].GDP**i
                        for i in range(1, degree+1)]).transpose(1, 0).reshape((-1, degree))
    
    #Step 2:Define train_y, each row of it represents a year of Life expectancy of Afghanistan. The Year column should be used to select the samples.
    train_y = np.array(total_df[total_df.Year <= 2013]['Life expectancy '])
    
    #Step 3:Define a LinearRegression model, and fit it using train_X and train_y.
    model = LinearRegression()
    model.fit(train_x, train_y)
    
    #Step 4:Calculate rmse and r2_score using the fitted model.
    train_rmse = mean_squared_error(model.predict(train_x), train_y)**0.5
    train_r2_score = model.score(train_x, train_y)
    
    '''end of student code'''
    print(f'Train set, degree={degree}, RMSE={train_rmse:.3f}, R2={train_r2_score:.3f}')
    #feel free to change the variable name if needed. 
    #DO NOT change the output format.
    '''Write your code here'''
    
    #Step 1: Define test_x and test_y by selecting the remaining years of the data
    test_x = np.array([total_df[total_df.Year > 2013].GDP**1
                        for i in range(1, degree+1)]).transpose(1, 0).reshape((-1, degree))
    test_y = np.array(total_df[total_df.Year > 2013]['Life expectancy '])
    
    #Step 2: Use model.predict to generate the prediction
    test_predict = model.predict(test_x)
    
    #Step 3: Calculate rmse and r2_score on test_x and test_y.
    test_rmse = mean_squared_error(test_predict, test_y)**0.5
    test_r2_score = model.score(test_x, test_y)

    '''end of student code'''
    print(f'Test set,  degree={degree}, RMSE={test_rmse:.3f}, R2={test_r2_score:.3f}')


# In[ ]:




