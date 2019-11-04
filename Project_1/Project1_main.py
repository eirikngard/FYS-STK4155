# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 18:56:16 2019

@author: Eirik Nordgård
"""
'''''''''''''''
Importing needed packages
'''''''''''''''

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
from sklearn.utils import shuffle
import seaborn as sns

#%%

'''''''''''''''''''''''''''''''''
DEFINING FUNCTIONS
'''''''''''''''''''''''''''''''''
  
def mse(data, model):
    """
    Calculates the mean square error between data and model.
    """
    #Fra slide 19
    n = np.size(model)
    mserror = np.sum((data - model)**2)/n 
    return mserror

def r2(data, model):
    """
    Calculates the R2-value of the model.
    """
    #Fra slide 19
    r2 = 1-np.sum((data-model)**2)/np.sum((data-np.mean(data))**2) 
    return r2

def bias(data, model):
    """caluclate bias from k expectation values and data of length n
    """
    #Fra slide 96
    n = np.size(model)
    bias = np.sum(data - np.mean(model)**2)/n
    return bias
        #The bias (or bias function) of an estimator is the difference 
        #between this estimator's expected value and the true value of 
        #the parameter being estimated.

def variance(data):
    """
    Calculating the variance of the model: Var[model]
    """
    
    #From slide 62
    n = np.size(data)
    variance = np.sum((data - np.mean(data))**2)/n
    return variance

#%% 
'''''''''''''''''''''''''''''''''''''''
Plotting the Francke function  
'''''''''''''''''''''''''''''''''''''''
fig = plt.figure()
ax = fig.gca(projection='3d')
# Make data.
n=20
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
noise=10*np.random.randn(n,n)
x, y = np.meshgrid(x,y)
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 #+ noise
z = FrankeFunction(x, y)

# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
color=fig.colorbar(surf, shrink=0.5, aspect=5)
#color.set_label('z', rotation=270)
plt.xlabel('x'); plt.ylabel('y');plt.title('Franke Function')
plt.show()

#2D figure
fig = plt.figure();plt.title('FrankeFunction 2D')
color=fig.colorbar(surf, shrink=0.5, aspect=5)
plt.imshow(z, cmap='coolwarm');plt.xlabel('X');plt.ylabel('Y');plt.show()
#%%
'''''''''''''''''''''''''''''''''''
Creating the Design Matrix 
'''''''''''''''''''''''''''''''''''

def CreateDesignMatrix_X(x, y, n = 5):
	"""
	Function for creating a design X-matrix with rows 
    [1, x, y, x^2, xy, xy^2 , etc.] Input is x and y mesh or raveled mesh, 
    keyword agruments n is the degree of the polynomial you want to fit.
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
z2 = z.flatten()         #Changing the dimentions of x,y,z from 20,20 to 400,1
x2 = x.flatten()
y2 = y.flatten()

X=CreateDesignMatrix_X(x2,y2,5)

#%% 
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
PERFORMING LINEAR REGRESSION (OLS, RIDGE and LASSO) ON THE FRANKEFUNCTION
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#Ordinary Least Squares (own) 
nrows,ncols=X.shape 
identity=np.eye(ncols)
B_ols=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z.ravel())  
y_tilde_ols= X @ B_ols#prediction, could also write y_tilde=X.dot(B_ols)

#Ridge regression (own)
lambd=0.001
nrows,ncols=X.shape 
identity=np.eye(ncols)
B_ridge=np.linalg.inv(X.T.dot(X)-lambd*identity).dot(X.T).dot(z2)
y_tilde_ridge = X @ B_ridge #y_tilde_ridge=X.dot(B_r)

#Lasso Regression (using scikitlearn)
_lambda = 0.001
clf_lasso = skl.Lasso(alpha=_lambda)#
clf_lasso.fit(X,z2)#Calculates Beta  
ylasso = clf_lasso.predict(X)  
B_lasso = clf_lasso.coef_ #B0,B1 etc Coefs. of Beta 
intercept = clf_lasso.intercept_ #Beta0
y_tilde_lasso_scikit=ylasso.ravel()

#Ordinary Least Squares (scikit-learn)
clf_linreg=skl.LinearRegression()
clf_linreg.fit(X,z2)
ylinreg=clf_linreg.predict(X)#Prediction
coefs=clf_linreg.coef_
intercept=clf_linreg.intercept_
y_tilde_ols_scikit = ylinreg.ravel()
                                 
#Ridge regression (scikit-lear)                       
_lambda = 0.01
clf_ridge = skl.Ridge(alpha=_lambda)#
clf_ridge.fit(X,z2)#Calculate B
yridge = clf_ridge.predict(X) #Prediction
coefs = clf_ridge.coef_ #B0,B1 etc Coefs. of Beta 
intercept = clf_ridge.intercept_ #Beta0
y_tilde_ridge_scikit=yridge.ravel()

#%%
'''''''''''''''''''''''''''''''''''''''
PLOTTING THE DEVIATION FROM THE FRANKEFUNCTION 
'''''''''''''''''''''''''''''''''''''''
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, y_tilde_ols.reshape(20,20)-z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel('x');plt.ylabel('y')
plt.title('OLS model deviation from Franke Function')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, y_tilde_ridge.reshape(20,20)-z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel('x');plt.ylabel('y')
plt.title('RIDGE model deviation from Franke Function')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, y_tilde_lasso_scikit.reshape(20,20)-z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel('x');plt.ylabel('y')
plt.title('LASSO model deviation from Franke Function')
plt.show()
#%%
'''''''''''''''''''''''''''''
Resampling the data using train_test_split 
'''''''''''''''''''''''''''''
#Split the data in test and training data
X_train, X_test, y_train, y_test = train_test_split(X, z2, test_size=0.2)
# matrix inversion to find beta
lambd=0.01
#OLS scikit (for comparison)
clf_linreg_train=skl.LinearRegression()
clf_linreg_train.fit(X_train,y_train)
y_train_predict = clf_linreg_train.predict(X_train)#lin_model.predict(X_train)
y_test_predict = clf_linreg_train.predict(X_test)

#LASSO scikit
clf_lasso = skl.Lasso(alpha=lambd)#
clf_lasso.fit(X_train,y_train)#Calculates Beta  
y_tilde_lasso_train = clf_lasso.predict(X_train).ravel() 
y_tilde_lasso_test = clf_lasso.predict(X_test).ravel()

#OLS own
B_ols_train=np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)  
y_tilde_ols_train= X_train @ B_ols_train
B_ols_test=np.linalg.inv(X_test.T.dot(X_test)).dot(X_test.T).dot(y_test)  
y_tilde_ols_test= X_test @ B_ols_test

#Ridge own 
B_ridge_train=np.linalg.inv(X_train.T.dot(X_train)-lambd*identity).dot(X_train.T).dot(y_train)
y_tilde_ridge_train = X_train @ B_ridge_train #y_tilde_ridge=X.dot(B_r)
B_ridge_test=np.linalg.inv(X_test.T.dot(X_test)-lambd*identity).dot(X_test.T).dot(y_test)
y_tilde_ridge_test = X_test @ B_ridge_test #y_tilde_ridge=X.dot(B_r)


# model evaluation for training set
mse_train=mean_squared_error(y_train,y_train_predict)
mse_train_ols=mse(y_train,y_tilde_ols_train)
mse_train_ridge=mse(y_train,y_tilde_ridge_train)
mse_train_lasso=mse(y_train,y_tilde_lasso_train)
r2_train = r2_score(y_train, y_train_predict)
#r2_train_ols=r2(y_train,y_tilde_ols_train)     #For some reason this wont work..
#r2_train_ridge=r2(y_train,y_tilde_ridge_train)
#r2_train_lasso = r2(y_train,y_tilde_lasso_train)

# model evaluation for testing set
mse_test=mean_squared_error(y_test,y_test_predict)
mse_test_ols=mse(y_test,y_tilde_ols_test)
mse_test_ridge=mse(y_test,y_tilde_ridge_test)
mse_test_lasso=mse(y_test,y_tilde_lasso_test)
r2_test = r2_score(y_test, y_test_predict)
#r2_test_ols=r2(y_test,y_tilde_ols_test)
#r2_test_ridge=r2(y_test,y_tilde_ridge_test)
#r2_test_lasso=r2(y_test,y_tilde_lasso_test)

print("The model performance for training set")
print("--------------------------------------")
print(f'MSE(scikit ols) is {mse_train:.8f}')
print(f'MSE(ols) is   {mse_train_ols:.8f}')
print(f'MSE(ridge) is {mse_train_ridge:.8f}')
print(f'MSE(lasso) is      {mse_train_lasso:.8f}')
print(f'R2 (ols) is {r2_train:.8f}')
#print(f'R2 (ridge) is {r2_test_ridge:.8f}')
#print(f'R2 (lasso) is {r2_test_lasso:.8f}')

print("The model performance for testing set")
print("--------------------------------------")
print(f'MSE(scikit ols) is  {mse_test:.8f}')
print(f'MSE (own, ols) is   {mse_test_ols:.8f}')
print(f'MSE (own, ridge) is {mse_test_ridge:.8f}')
print(f'MSE (lasso) is      {mse_test_lasso:.8f}')
print(f'R2 score is  {r2_test:.8f}')
#%%
'''''''''''''''''''''''''''''''''''''''
plotting the y_test vs y_test_pred
'''''''''''''''''''''''''''''''''''''''
# Should have been a straight line in a perfect world
fig = plt.figure()
plt.scatter(y_test, y_test_predict)
plt.title("Scatterplot of y and y_pred of test-set")
plt.xlabel("y_test")
plt.ylabel("y_test_predict")
plt.show()

    
#%%
'''''''''''''''''''''''''''''
CONFIDENCE INTERVALL FOR BETAS
'''''''''''''''''''''''''''''
ols_conf=[]
ridge_conf=[]
lasso_conf=[]
    
def confidence(beta):
    weight = np.sqrt( np.diag( np.linalg.inv( X.T @ X ) ) )*1.96
    betamin = beta - weight
    betamax = beta + weight
    return betamin, betamax

olsconf=np.array(confidence(B_ols))
olsmin=olsconf[0,:] 
olsmax=olsconf[1,:]
plt.figure(); plt.plot(B_ols,label=r'$\beta$'" "'OLS') 
plt.plot(olsmin,'k--', linewidth=2, markersize=1,label=r'$\beta$'" "'min')
plt.plot(olsmax,'r--', linewidth=2, markersize=1, label=r'$\beta$'" "'max')
plt.legend();plt.title('Confidence interval for' r'$\beta$'" "'OLS');plt.show()


ridgeconf=np.array(confidence(B_ridge))
ridgemin=ridgeconf[0,:] 
ridgemax=ridgeconf[1,:]
ridgeconf=confidence(B_ridge)
ridgemin=ridgeconf[0];ridgemax=ridgeconf[1]
plt.figure();plt.plot(B_ridge,label=r'$\beta$'" "'RIDGE')
plt.plot(ridgemin,'k--', linewidth=2, markersize=1,label=r'$\beta$'" "'min')
plt.plot(ridgemax,'r--', linewidth=2, markersize=1,label=r'$\beta$'" "'max')
plt.legend();plt.title('Confidence interval for' r'$\beta$'" "'RIDGE');plt.show()

lassoconf=np.array(confidence(B_lasso))
lassomin=lassoconf[0,:] 
lassomax=lassoconf[1,:]
lassoconf=confidence(B_lasso)
lassomin=lassoconf[0];lassomax=lassoconf[1]
plt.figure();plt.plot(B_lasso,label=r'$\beta$'" "'LASSO')
plt.plot(lassomin,'k--', linewidth=2, markersize=1,label=r'$\beta$'" "'min')
plt.plot(lassomax,'r--', linewidth=2, markersize=1,label=r'$\beta$'" "'max')
plt.legend();plt.title('Confidence interval for' r'$\beta$'" "'LASSO');plt.show()

#Decreace lambda for B_lasso and it wil look different 

#%%
'''''''''''''''''''''''''''
Calculating MSE as a function of lambda (ridge)
'''''''''''''''''''''''''''

x=np.random.random(n)#Disse er for å finne optimal lambda
#x = np.arange(0, 1, 0.05)
y=np.random.random(n)#Disse er for å finne otimal lambda
#y = np.arange(0, 1, 0.05)

lambdaas = np.logspace(-8,0,10) #np.arange(0.00001,1.00001,1e-2) 

msee=np.zeros((5,len(lambdaas)))
msee_sk=np.zeros((5,len(lambdaas)))
R2=np.zeros((5,len(lambdaas)))
R2_sk=np.zeros((5,len(lambdaas)))

for n in range(1,6):
    X=CreateDesignMatrix_X(x2,y2,n)
    nrows,ncols=X.shape
    identity=np.eye(ncols)
    beta_n=[]

    for i,lambdaa in enumerate(lambdaas): 
        
        y_tilde_ridge_tests=skl.Ridge(alpha=lambdaa).fit(X,z2).predict(X)
        B_ridge_test=np.linalg.inv(X.T.dot(X)-lambdaa*identity).dot(X.T).dot(z2)
        
        y_tilde_ridge_test = X @ B_ridge_test
        msee[n-1,i]=mse(z2,y_tilde_ridge_test)
        if mse(z2,y_tilde_ridge_test)>20:
            print(np.linalg.det(X.T.dot(X)-lambdaa*identity))
            #print(np.linalg.det(X.T.dot(X)-lambdaas[87]*identity))
            #print(np.linalg.det(X.T.dot(X)-lambdaas[89]*identity))
            print(lambdaa)
            print(n)
            print(np.where(lambdaas==lambdaa))
        msee_sk[n-1,i]=mse(z2,y_tilde_ridge_tests)
        R2[n-1,i]=(r2(z2,y_tilde_ridge_test)) 
        #The line above is ocationally causing problems, dont know why
        R2_sk[n-1,i]=(r2(z2,y_tilde_ridge_tests))
        
print(min(msee_sk[4,:]))

#%%

plt.figure()
plt.plot(np.log10(lambdaas),msee[4,:],label='MSE')
plt.plot(np.log10(lambdaas),msee_sk[4,:],label='MSE scikit-learn')   
plt.xlabel('log scale');plt.ylabel('MSE');plt.title('MSE as a function of'" " r'$\lambda$') 
plt.legend()
plt.figure()
plt.plot(np.log10(lambdaas),R2[4,:],label='R2')
plt.plot(np.log10(lambdaas),R2_sk[4,:],label='R2 scikit-learn')
plt.xlabel('log scale');plt.ylabel('R2 score');plt.title('R2 as a function of'" " r'$\lambda$') 
plt.legend()

#%%
'''''''''''''''''''''''''''''''''
Printing results for OLS, RIDGE 
and LASSO regression on the FrankeFunction.
'''''''''''''''''''''''''''''''''

#Restate original x nd y

x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)


print("--------------------------------------")
print("Ordinary Least Squares")
print("--------------------------------------")
print("MSE (own code): %.8f" % mse(z2,y_tilde_ols))
print("MSE (scikit-learn): %.8f" % mse(z2,y_tilde_ols_scikit))
print("R2 (own code): %.8f" % r2(z2,y_tilde_ols))
print("R2 (scikit-learn): %.8f" % r2(z2,y_tilde_ols_scikit))
print("BIAS (own code): %.8f" % bias(z2,y_tilde_ols))
print("BIAS (scikit-learn): %.8f" % bias(z2,y_tilde_ols_scikit))
print("VARIANCE (own code): %.8f" % variance(z2))

print("--------------------------------------")
print("RIDGE Regression")
print("--------------------------------------")
print("MSE (own code): %.8f" % mse(z2,y_tilde_ridge))
print("MSE (scikit-learn): %.8f" % mse(z2,y_tilde_ridge_scikit))
print("R2 (own code): %.8f" % r2(z2,y_tilde_ridge))
print("R2 (scikit-learn): %.8f" % r2(z2,y_tilde_ridge_scikit))
print("BIAS (own code): %.8f" % bias(z2,y_tilde_ridge))
print("BIAS (scikit-learn): %.8f" % bias(z2,y_tilde_ridge_scikit))
print("VARIANCE (own code): %.8f" % variance(z2))

print("--------------------------------------")
print("LASSO Regression")
print("--------------------------------------")
print("MSE (scikit-learn): %.8f" % mse(z2,y_tilde_lasso_scikit))
print("R2 (scikit-learn): %.8f" % r2(z2,y_tilde_lasso_scikit))
print("BIAS (scikit-learn): %.8f" % bias(z2,y_tilde_lasso_scikit))
print("VARIANCE (own code): %.8f" % variance(z2))


#%%
'''''''''''''''''''''''''''''''''''''''
PLOTTING MSE AS A FUNCTION OF COMPLEXITY
'''''''''''''''''''''''''''''''''''''''
n=np.arange(10)
error_ols = []
error_ridge=[]
error_lasso=[]
identity=np.eye(ncols)
lam=np.logspace(-8,2,10)
lambd=0.01
#for lambd in lam:
for i in n:
    X=CreateDesignMatrix_X(x2,y2,i)
            
    B_ols=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z.ravel())  
    y_tilde_ols= X @ B_ols#prediction
    
    #If this doens not run, comment out the next line, re-run section, then put it back and run
    B_ridge=np.linalg.inv(X.T.dot(X)-lambd*identity).dot(X.T).dot(z2)
    nrows,ncols=X.shape 
    
    y_tilde_ridge = X @ B_ridge 
    y_tilde_lasso=skl.Lasso(alpha=lambd).fit(X,z2).predict(X)
            
    error_OLS=mse(z2,y_tilde_ols)
    error_ols.append(error_OLS)
    error_RIDGE=mse(z2,y_tilde_ridge)
    error_ridge.append(error_RIDGE)
    error_LASSO=mse(z2,y_tilde_lasso)
    error_lasso.append(error_LASSO)
    
plt.figure()
plt.plot(n,error_ols,'k--',lw=3,dashes=(5,1),label="MSE OLS")
plt.plot(n,error_ridge,'b--',lw=3,dashes=(5,5),label="MSE RIDGE")
plt.plot(n,error_lasso,'r--',lw=3,dashes=(5,5),label="MSE LASSO")
plt.title("MSE as a function of model complexity")
plt.xlabel("Model Complexity (Polynomial Degree)");plt.ylabel("MSE")
plt.legend();plt.show()    
#%%
lam=np.logspace(-3,0,10)
error_lasso=[]
plt.figure()
for lambd in lam:
    
    
    X=CreateDesignMatrix_X(x2,y2,5)
    y_tilde_lasso=skl.Lasso(alpha=lambd).fit(X,z2).predict(X)
    error_LASSO=mse(z2,y_tilde_lasso)
    error_lasso.append(error_LASSO)

plt.plot(lam,error_lasso,'r--',lw=3,dashes=(5,5),label="MSE LASSO")
plt.title("MSE LASSO as function og lambda")
plt.xlabel("Model Complexity (Polynomial Degree)");plt.ylabel("MSE")
plt.legend();plt.show()   
    
#%%    
'''''''''
Plotting Terrain Data 2D
'''''''''   
     
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# Load the terrain
terrain1 = imread('SRTM_data_Norway_1.tif') #Data
# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain1, cmap='coolwarm')
plt.xlabel('X'); plt.ylabel('Y'); plt.show()
#%%
'''''''''
Plotting Terrain Data 3D
'''''''''
xt = np.arange(0,1801,1)
yt = np.arange(0,3601,1)
xt, yt = np.meshgrid(xt,yt)
zt = terrain1; zt2 = terrain1.ravel()
fig = plt.figure(); ax = fig.gca(projection='3d')
surf = ax.plot_surface(xt, yt, zt, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Terrain over Norway 1'); plt.xlabel('X'); plt.ylabel('Y') ;plt.show()
#%%
'''''''''
From her and down is not used, code not wrking properly. 
'''''''''
'''''''''''''''''''''
Performing OLS regression on the Terrain data and making a fit
'''''''''''''''''''''

#Creating Desig Matrix
Xt=CreateDesignMatrix_X(xt,yt,5)

#Ordinary Least Squares, Own code 
nrows,ncols=Xt.shape #Gjøres for å få riktig I matrise under
identity=np.eye(ncols)
B_ols_t=np.linalg.inv(Xt.T.dot(Xt)).dot(Xt.T).dot(zt.ravel())#fit y.ravel og y.flatten er det samme  
yt_tilde_ols= Xt @ B_ols_t#prediction, could also write y_tilde=X.dot(B_ols)
#yt_tilde_ols you should plot
yt2_own=yt_tilde_ols.reshape(3601,1801)
#yt2_andres=least_squares(Xt,zt.ravel(),0).reshape(3601,1801)

#%%
print("--------------------------------------")
print("Ordinary Least Squares")
print("--------------------------------------")
print("MSE own code Franke: %.8f" % mse(z2,y_tilde_ols))
print("MSE own code Terr: %.8f" % mse(zt,yt2_own))


print("R2 oc Franke: %.8f" % r2(z2,y_tilde_ols))
print("R2 oc Terr: %.8f" % r2(zt,yt2_own))

print("BIAS oc Franke: %.8f" % bias(z2,y_tilde_ols))
print("BIAS oc Terr: %.8f" % bias(zt,yt2_own))

print("VARIANCE oc Franke: %.8f" % variance(z2))
print("VARIANCE oc Terr: %.8f" % variance(zt))


