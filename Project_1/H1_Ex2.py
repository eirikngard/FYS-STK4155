# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 08:31:54 2019

@author: Eirik Nordgård

Homework1, Ex.2
"""
################################### 1

import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import pandas as pd
import sklearn.linear_model as skl

#Computing the parametrization of the data set 
#fitting a second-order polynomial 

x=np.random.rand(100,1)
y=5*x*x*0.5*np.random.randn(100,1)

linreg=LinearRegression()
linreg.fit(x,y,2)
ypredict=linreg.predict(x)

plt.plot(x,y,'ro')
plt.plot(x,ypredict,"bo")
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Linear Regression Fit')
plt.show()

#%%
# OR LIKE THIS 

import numpy as np
import matplotlib.pyplot as plt
x = np.random.rand(100)
y = 5*x*x+0.1*np.random.randn(100)
print(y.T)
p = np.poly1d(np.polyfit(x.T, y,2))
plt.plot(x, y, '-')
plt.show()

# Design matrix
X=np.c_[np.ones((100,1)),x,x**2]

clf_linreg=skl.LinearRegression()
clf_linreg.fit(X,y)
ylinreg=clf_linreg.predict(X)
coefs=clf_linreg.coef
intercept=clf_linreg.intercept

plt.plot(x,y,'ro')
plt.plot(x,ypredict,"bo")
plt.plot(x,ylinreg,"go")
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Linear Regression Fit')
plt.show()

#%%
############################### 2
##Using scikit-learn

#lr = LogisticRegression()
#lr.fit(train_dataset,train_labels)

#nsamples, nx, ny = train_dataset.shape
#d2_train_dataset = train_dataset.reshape((nsamples,nx*ny))

'''
linreg=LinearRegression()
linreg.fit(x,y)
nsamples,nx,ny =x.shape
d2_x =x.reshape((nsamples,nx*ny))

#scikit-learn expects 2d num arrays for the training dataset for a fit 
#function. The dataset you are passing in is a 3d array you need to 
#reshape the array into a 2d.

noise = np.asarray(random.sample((range(200)),200))
y=x**3
yn=x**3*100
poly3 = PolynomialFeatures(degree=3)
X = poly3.fit_transform(x[:,np.newaxis])
clf3 = LinearRegression()
clf3.fit(X,y)

Xplot=poly3.fit_transform(x[:,np.newaxis])
poly3_plot=plt.plot(x, clf3.predict(Xplot), label='Cubic Fit')
plt.plot(x,yn, color='red', label="True Cubic")
plt.scatter(x, y, label='Data', color='orange', s=15)
plt.legend()
plt.show()

#%% C
print('Mean Squared Error (MSE): %.3f' % mean_squared_error(y,ypredict))
print('Variance Score: %.3f'% r2_score(y,ypredict))

'''
#Comment 