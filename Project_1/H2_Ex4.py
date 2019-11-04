# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 08:32:22 2019

@author: Eirik Nordgård 
Homewrok 2, Ex. 4-5
"""

#Helpfull pacages
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import pandas as pd
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#%%
# Exercise 4.1 

#Ridge Regression

#Shrinkidge method where you do a regression, adjusting lambda to minimize the MSE
x=np.random.rand(100,1)
y=5*x*x+0.1*np.random.randn(100,1)
#%%
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

#%% Exercise 4.2   Using scikit-learn 

_lambda = 0.1
clf_ridge = skl.Ridge(alpha=_lambda)#
clf_ridge.fit(X,y)#Calculates Beta in line 35 
yridge = clf_ridge.predict(X) #Same as y_pred in line 40
coefs = clf_ridge.coef_ #B0,B1 etc Coefs. of Beta 
intercept = clf_ridge.intercept_ #Gir Beta0

plt.plot(x,y,'ro',label='y')
plt.plot(x,y_pred,"bo",label='y_ringe_analyticalmodel')
plt.plot(x,yridge,"go",label='y_ridge_scikit.earn')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r' Ridge Regression Fit')
plt.legend() 
plt.show()

# The mean squared error      
print("Mean squared error: %.5f" % mean_squared_error(y, yridge))

#%% Exercise 4.3 Finding the variance of the Betas

print(coefs) #Coefs is the coefs of B, so B0, B1, B2 osv
B_r2=np.linalg.inv(X.T.dot(X)+lambd*identity) 
var=np.diagonal(B_r2)#Definisjon av variansen til Betaene fra oppgavesett

#var er variansen til alle betaene, i dette tilfelle 3 stykker

# SE HVORDAN VARIANSEN ENDRER SEG HVIS DU ENDRER LAMBDA

#%% Exercise 4.4 Doing the same as above, but with Lasso regression 

_lambda = 0.01
clf_lasso = skl.Lasso(alpha=_lambda)#
clf_lasso.fit(X,y)#Calculates Beta in line 35 
ylasso = clf_lasso.predict(X) #Same as y_pred in line 40
coefs = clf_lasso.coef_ #B0,B1 etc Coefs. of Beta 
intercept = clf_lasso.intercept_ #Gir Beta0

plt.plot(x,y,'ro',label='y')
plt.plot(x,y_pred,"bo",label='y_ridge')
plt.plot(x,ylasso,"go",label='y_lasso')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r' Lasso Regression Fit')
plt.legend() 
plt.show()

#Fitten er dårligere for Lasso enn Rindge for en lik forandring
#av lambda.  

#%% Exercise 4.5

#Mean squared error a
print("Mean Squared Error Ridge: %.3f" % mean_squared_error(y,yridge))
print("Mean Squared Error Lasso: %.3f" % mean_squared_error(y,ylasso))
#R2 score: Explained variance score: 1 i perfect prediction
print("Variance score Ridge: %.f3" % r2_score(y,yridge))
print("Variance score Lasso: %.f3" % r2_score(y,ylasso))
#Hvorfor gir disse to siste output 13??? 

def r2_score_ridge(y,yridge):
    y_hat=np.sum(y)/len(y)
    return 1-(np.sum(y**2-yridge**2)/np.sum((y**2-y_hat**2))) #Definisjon av R2

def r2_score_lasso(y,ylasso):
    y_hat=np.sum(y)/len(y)
    return 1-(np.sum(y**2-ylasso**2)/np.sum((y**2-y_hat**2))) #Definisjon av R2

def mse_ridge(y,yridge):
    return np.sum(y**2-yridge**2)/len(y) #Definisjon av MSE
    
def mse_lasso(y,ylasso):
    return np.sum(y**2-ylasso**2)/len(y) #Definisjon av MSE
    

MSE_Ridge=mse(y,yridge)
MSE_Lasso=mse(y,ylasso) 
R2_score_Ridge=r2_score(y,yridge)
R2_score_Lasso=r2_score(y,ylasso)
#FEIL Her er begge R2 score like. Hvorfor? 

'''
# Alternativ måte å regne ut R2, fra lecturenotes
def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

print(MSE(Energies,ytilde))

def RelativeError(y_data,y_model):
    return abs((y_data-y_model)/y_data)
print(RelativeError(Energies, ytilde))
'''

