# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 23:18:45 2018

@author: Akash
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Fitting Linear regression
from sklearn.linear_model import LinearRegression 
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitiing Polynomial regression 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5) 
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

#visualisation for Linear regression
plt.scatter(X, y, color = 'green')
plt.plot(X, lin_reg.predict(X), color = 'red')
plt.title('Truth or Bluff (Linear regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#visualisation for Polynomial regression
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'green')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = 'red')
plt.title('Truth or Bluff (Polynomial regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#Prediciting a new result using Linear regrssion
a = int(input('Enter Previous Level: '))
print("Estimated salary will be: " , lin_reg.predict(a))

#Prediciting a new result using Polynomial regrssion
a = int(input('Enter Previous Level: '))
print("Estimated salary will be: " ,lin_reg2.predict(poly_reg.fit_transform(a)))








