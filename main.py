# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 21:35:37 2018

@author: foyaz
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt

dataset = pd.read_csv('apartment_data.csv')


#print(dataset.head())

X = dataset.iloc[:, 1:13].values  
Y = dataset.iloc[:, -1].values 

from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0) 

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, Y_train) 


Y_pred = regressor.predict(X_test) 

df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})  

#print(df) 


cnt=0
acc=0.0

for (x,y) in zip(Y_test,Y_pred):
    cnt = cnt + 1
    acc = acc + abs(x-y)/y

acc = acc * 100
acc = acc / cnt

acc = 100 - acc 

print('Accuracy : ',acc)


    
