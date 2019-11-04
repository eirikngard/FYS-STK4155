# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:53:00 2019

@author: Eirik Nordgård
"""
'''
RIDGE REGRESSION
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

## VIKTIG å fjerne frste kolonne med 1 for sammenlikning 

#%% 
'''
TASK D: RIDGE REGRESSION  
'''         

#Using own code 
lambd=0.019
X=np.c_[np.ones((100,1)),x,x**2]
nrows,ncols=X.shape #Gjøres for å få riktig I matrise under
identity=np.eye(ncols)
B_r=np.linalg.inv(X.T.dot(X)-lambd*identity).dot(X.T).dot(y)
# Likning 3.44 i The Elements of Statistical Learning

def mse(y,y_pred):
    return np.sum(y**2-y_pred**2)/len(y) #Definisjon av MSE
    
y_pred=X.dot(B_r)

result=mse(y,y_pred) #Dette gir scoren min for prediksjonen, så nære 0 som mulig 

#Plotter y og y_pred 
plt.plot(x,y,'ro',label='y')
plt.plot(x,y_pred,"bo",label='y_pred')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r' Ridge Regression Fit')
plt.legend()
plt.show()
print('Mean Squared Error (MSE): %.5f' % result)

#Using scikit-lear                       
_lambda = 0.01
clf_ridge = skl.Ridge(alpha=_lambda)#
clf_ridge.fit(X,z2)#Calculate B
yridge = clf_ridge.predict(X) #Same as y_pred in line 40
coefs = clf_ridge.coef_ #B0,B1 etc Coefs. of Beta 
intercept = clf_ridge.intercept_ #Gir Beta0
ylinreg2=yridge.reshape(20,20)
'''
For lam=100: MSE=0,03726 R2=-0,75278
For lam10: MSE=0,02322 R2=0,49920
For lam=1: MSE=0,01509 R2=0,74616
For lam=.1: MSE=0,00957 R2=0,85900
For lam=.01: MSE=0,00651 R2=0,90995
For lam=.001: MSE= 0,00386 R2=0,94920
For lam=.0001: MSE=0,00265 R2=0,96608
For lam=0.00001: MSE= 0,00223 R2=0,97188
'''
#%%
# Plotting how the surface of how the Lin.Reg deviates from Franke function

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, ylinreg2-z, cmap=cm.coolwarm,
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
print("Mean squared error: %.5f" % mean_squared_error(yridge, z2))
# The R2 score      
print("Variance score: %.5f" % r2_score(yridge, z2))

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
Task B: Resampling techniques and adding complexity, splitting in train and test
'''
##k-fold (from slides)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# A seed just to ensure that the random numbers are the same for every run.
# Useful for eventual debugging.
np.random.seed(80)
# Generate the data.
nsamples = 100
#x = np.random.randn(nsamples)
#y = 3*x**2 + np.random.randn(nsamples)

## Cross-validation on Ridge regression using KFold only

# Decide degree on polynomial to fit
poly = PolynomialFeatures(degree = 5)

# Decide which values of lambda to use
nlambdas = 500
lambdas = np.logspace(-3, 5, nlambdas)#Numbers spaced on log sale

# Initialize a KFold instance
k = 5
kfold = KFold(n_splits = k) #create k folds which data can be splitted into

# Perform the cross-validation to estimate MSE
scores_KFold = np.zeros((nlambdas, k)) 

i = 0
for lmb in lambdas:
    ridge = Ridge(alpha = lmb)
    j = 0
    for train_inds, test_inds in kfold.split(z2):
        xtrain = x[train_inds]
        ytrain = y[train_inds]

        xtest = x[test_inds]
        ytest = y[test_inds]

        Xtrain = poly.fit_transform(xtrain[:, np.newaxis])
        ridge.fit(Xtrain, ytrain[:, np.newaxis])

        Xtest = poly.fit_transform(xtest[:, np.newaxis])
        ypred = ridge.predict(Xtest)

        scores_KFold[i,j] = np.sum((ypred - ytest[:, np.newaxis])**2)/np.size(ypred)

        j += 1
    i += 1


estimated_mse_KFold = np.mean(scores_KFold, axis = 1)

## Cross-validation using cross_val_score from sklearn along with KFold

# kfold is an instance initialized above as:
# kfold = KFold(n_splits = k)

estimated_mse_sklearn = np.zeros(nlambdas)
i = 0
for lmb in lambdas:
    ridge = Ridge(alpha = lmb)

    X = poly.fit_transform(x[:, np.newaxis])
    estimated_mse_folds = cross_val_score(ridge, X, y[:, np.newaxis], scoring='neg_mean_squared_error', cv=kfold)

    # cross_val_score return an array containing the estimated negative mse for every fold.
    # we have to the the mean of every array in order to get an estimate of the mse of the model
    estimated_mse_sklearn[i] = np.mean(-estimated_mse_folds)

    i += 1

## Plot and compare the slightly different ways to perform cross-validation

plt.figure()

plt.plot(np.log10(lambdas), estimated_mse_sklearn, label = 'cross_val_score')
plt.plot(np.log10(lambdas), estimated_mse_KFold, 'r--', label = 'KFold')

plt.xlabel('log10(lambda)')
plt.ylabel('mse')

plt.legend()

plt.show()
'''
#%%
# We split the data in test and training data using sklearn

X_train, X_test, y_train, y_test = train_test_split(X, z2, test_size=0.2)
# matrix inversion to find beta

clf_ridge_train=skl.Ridge(alpha=_lambda)
clf_ridge_train.fit(X_train,y_train)
#%%
# model evaluation for training set

y_train_predict = clf_ridge_train.predict(X_train)#lin_model.predict(X_train)
#rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
mse_train=MSE(y_train,y_train_predict)
r2 = r2_score(y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print(f'MSE is {mse_train:.8f}')
print(f'R2 score is {r2:.8f}')
#%%
# model evaluation for testing set

y_test_predict = clf_ridge_train.predict(X_test)
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
