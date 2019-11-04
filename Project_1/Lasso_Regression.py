# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:13:45 2019

@author: Eirik N
"""

#%%

'''
#%% LASSO REGRESSION

_lambda = 0.01
clf_lasso = skl.Lasso(alpha=_lambda)#
clf_lasso.fit(X,y)#Calculates Beta in line 35 
ylasso = clf_lasso.predict(X) #Same as y_pred in line 40
coefs = clf_lasso.coef_ #B0,B1 etc Coefs. of Beta 
intercept = clf_lasso.intercept_ #Gir Beta0

'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

import random
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import pandas as pd
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#%% 
''' 
Plotting the Francke function  
'''

fig = plt.figure()
ax = fig.gca(projection='3d')
# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4
z = FrankeFunction(x, y)
# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

#%%
'''
Creating the design matrix 
'''

def CreateDesignMatrix_X(x, y, n = 5):
	"""
	Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
	Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
	"""
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = x**(i-k) * y**k

	return X



z = FrankeFunction(x,y)
z2 = z.flatten()#Endrer dimesjonen på x,y,z fra 20,20 til 400,1
x2 = x.flatten()
y2 = y.flatten()

X=CreateDesignMatrix_X(x2,y2,5)

#%% 
'''
TASK D: LASSO REGRESSION
'''
                                
_lambda = 0.01
clf_lasso = skl.Lasso(alpha=_lambda)#
clf_lasso.fit(X,z2)#Calculates Beta in line 35 
ylasso = clf_lasso.predict(X) #Tilnærming til frankefunskjonen 
coefs = clf_lasso.coef_ #B0,B1 etc Coefs. of Beta 
intercept = clf_lasso.intercept_ #Gir Beta0
ylasso2=ylasso.reshape(20,20)
#Reshape av ylasso fra 400,1 til 20,20 så plotsurface kan lese inn ny ylasso

'''
TASK D: RIDGE REGRESSION  
                             
_lambda = 0.1
clf_ridge = skl.Ridge(alpha=_lambda)#
clf_ridge.fit(X,z2)#Calculate B
yridge = clf_ridge.predict(X) #Same as y_pred in line 40
coefs = clf_ridge.coef_ #B0,B1 etc Coefs. of Beta 
intercept = clf_ridge.intercept_ #Gir Beta0
ylinreg2=yridge.reshape(20,20)
'''
#%%
# Plotting how the surface of how the Lin.Reg deviates from Franke function

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, ylasso2-z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False) #Removing z gives plot for linreg.
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
#%%
'''
MSE and R2 score
'''
# The mean squared error      
print("Mean squared error: %.5f" % mean_squared_error(ylasso, z2))
# The R2 score      
print("Variance score: %.5f" % r2_score(ylasso, z2))

#Alternative way
def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_model)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

#Vil finne et polynom som på best mulig måte gjenskaper frankefunction
#Kommenter hva MSE og R2 er og hva de sier. Er mine verdier høye eller lave?

#%%
'''
Find confidence intervall of the Bs, using the variance of each paramtere
'''


#%%
'''
Task B: Resampling techniques and adding complexity
'''
#%%
# We split the data in test and training data using sklearn

X_train, X_test, y_train, y_test = train_test_split(X, z2, test_size=0.2)
# matrix inversion to find beta

clf_lasso_train=skl.Lasso(alpha=_lambda)
clf_lasso_train.fit(X_train,y_train)
#%%
# model evaluation for training set

y_train_predict = clf_lasso_train.predict(X_train)#lin_model.predict(X_train)
#rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
mse_train=MSE(y_train,y_train_predict)
r2 = r2_score(y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print(f'MSE is {mse_train:.8f}')
print(f'R2 score is {r2:.8f}')
#%%
# model evaluation for testing set

y_test_predict = clf_lasso_train.predict(X_test)
#rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
mse_test=MSE(y_test,y_test_predict)
r2 = r2_score(y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print(f'MSE is {mse_test:.8f}')
print(f'R2 score is {r2:.8f}')
#%%
# plotting the y_test vs y_pred
# ideally should have been a straight line
fig = plt.figure()
plt.scatter(y_test, y_test_predict)
plt.title("Scatterplot of y and y_pred of test-set")
plt.xlabel("y_test")
plt.ylabel("y_test_predict")
plt.show()

#%%
'''''''''
From Notes
'''''''''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

np.random.seed(2018)

n = 500
n_boostraps = 100
degree = 18  # A quite high value, just to show.
noise = 0.1

# Make data set.
x = np.linspace(-1, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)

# Hold out some test data that is never used in training.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Combine x transformation and model into one operation.
# Not neccesary, but convenient.
model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))

# The following (m x n_bootstraps) matrix holds the column vectors y_pred
# for each bootstrap iteration.
y_pred = np.empty((y_test.shape[0], n_boostraps))
for i in range(n_boostraps):
    x_, y_ = resample(x_train, y_train)

    # Evaluate the new model on the same test data each time.
    y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()

# Note: Expectations and variances taken w.r.t. different training
# data sets, hence the axis=1. Subsequent means are taken across the test data
# set in order to obtain a total value, but before this we have error/bias/variance
# calculated per data point in the test set.
# Note 2: The use of keepdims=True is important in the calculation of bias as this 
# maintains the column vector form. Dropping this yields very unexpected results.
error = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
bias = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
variance = np.mean( np.var(y_pred, axis=1, keepdims=True) )
print('Error:', error)
print('Bias^2:', bias)
print('Var:', variance)
print('{} >= {} + {} = {}'.format(error, bias, variance, bias+variance))

plt.plot(x[::5, :], y[::5, :], label='f(x)')
plt.scatter(x_test, y_test, label='Data points')
plt.scatter(x_test, np.mean(y_pred, axis=1), label='Pred')
plt.legend()
plt.show()
