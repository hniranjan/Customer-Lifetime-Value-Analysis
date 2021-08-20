#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df=pd.read_csv("WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv")

label_encoder = preprocessing.LabelEncoder() 

for i in df.columns:
    if df[i].dtype==object:
        df[i]=label_encoder.fit_transform(df[i])
        

x=df.drop('Customer Lifetime Value',axis=1)
y=df['Customer Lifetime Value']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


###########################Linear Regression#################################

from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()
lin_reg.fit(x_train,y_train)

y_pred=lin_reg.predict(x_test)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_pred)


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    z = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(z)
    return
mean_absolute_percentage_error(y_test,y_pred)

rmse=np.sqrt(mse)
print(rmse)


###########################Decision Tree#################################

from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(random_state=42)
dtr.fit(x_train,y_train)
y_pred1=dtr.predict(x_test)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    z = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(z)
    return
mean_absolute_percentage_error(y_test,y_pred1)




mse1=mean_squared_error(y_test,y_pred1)
rmse1=np.sqrt(mse1)


print(rmse1)

###########################Random Forest#################################

from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=1000,random_state=42)
rf_reg.fit(x_train,y_train)
y_pred2=rf_reg.predict(x_test)
mse2=mean_squared_error(y_test,y_pred2)
rmse2=np.sqrt(mse2)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    z = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(z)
    return
mean_absolute_percentage_error(y_test,y_pred2)

print(rmse2)

##########################Random Forest#################################
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=2000,random_state=42)
rf_reg.fit(x_train,y_train)
y_pred2=rf_reg.predict(x_test)
mse2=mean_squared_error(y_test,y_pred2)
rmse2=np.sqrt(mse2)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    z = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(z)
    return
mean_absolute_percentage_error(y_test,y_pred2)


print(rmse2)






