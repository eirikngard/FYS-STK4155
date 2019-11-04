# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:04:30 2019

@author: Eirik Nordgård 
"""

"""
Setup of file-reading on Taiwan credit information.
Code from Mortens logistic regression sides.

"""

import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

# Trying to set the seed
np.random.seed(0)
import random
random.seed(0)


def createx():
    
    # Reading file into data frame
    cwd = os.getcwd()
    filename = cwd + '/default of credit card clients.xls'
    nanDict = {}
    df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
    
    df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)
    
    df = df.drop(df[(df.BILL_AMT1 == 0) &
                (df.BILL_AMT2 == 0) &
                (df.BILL_AMT3 == 0) &
                (df.BILL_AMT4 == 0) &
                (df.BILL_AMT5 == 0) &
                (df.BILL_AMT6 == 0)].index)
    
    df = df.drop(df[(df.PAY_AMT1 == 0) &
                    (df.PAY_AMT2 == 0) &
                    (df.PAY_AMT3 == 0) &
                    (df.PAY_AMT4 == 0) &
                    (df.PAY_AMT5 == 0) &
                    (df.PAY_AMT6 == 0)].index)
    
    print(df.head())
    print(df.shape)
    # Features and targets 
    a = df.loc[:, df.columns != 'defaultPaymentNextMonth']
    print(a.head())
    
    X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
    #print(X.shape)
    y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values
    #print(y.shape)
    
    # Categorical variables to one-hot's
    onehotencoder = OneHotEncoder(categories="auto")
    
    X = ColumnTransformer(
        [("", onehotencoder, [3]),],
        remainder="passthrough"
    ).fit_transform(X)
    
    print(X.shape)
    
    y.shape
    
    # Train-test split
    trainingShare = 0.5 
    seed  = 1
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, train_size=trainingShare, \
                                                  test_size = 1-trainingShare,
                                                  random_state=seed)
    
    # Input Scaling
    sc = StandardScaler()
    XTrain = sc.fit_transform(XTrain)
    XTest = sc.transform(XTest)
        
    # One-hot's of the target vector
    #Y_train_onehot = onehotencoder.fit_transform(yTrain)
    #, Y_test_onehot = onehotencoder.fit_transform(yTest)
    
    
    # Remove instances with zeros only for past bill statements or paid amounts
    '''
    df = df.drop(df[(df.BILL_AMT1 == 0) &
                    (df.BILL_AMT2 == 0) &
                    (df.BILL_AMT3 == 0) &
                    (df.BILL_AMT4 == 0) &
                    (df.BILL_AMT5 == 0) &
                    (df.BILL_AMT6 == 0) &
                    (df.PAY_AMT1 == 0) &
                    (df.PAY_AMT2 == 0) &
                    (df.PAY_AMT3 == 0) &
                    (df.PAY_AMT4 == 0) &
                    (df.PAY_AMT5 == 0) &
                    (df.PAY_AMT6 == 0)].index)
    '''
 
    #return (XTrain.todense(), XTest.todense(), 
    #            Y_train_onehot.todense(), Y_test_onehot.todense() )
    return (XTrain, XTest, yTrain, yTest)
    #return (XTrain,yTrain)


#XTrain, XTest, yTrain, yTest = createx()
Xtrain,XTest,yTrain,yTest = createx()
#X=createx()

#%%

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

lambdas=np.logspace(-5,7,13)
parameters = [{'C': 1./lambdas, "solver":["lbfgs"]}]#*len(parameters)}]
scoring = ['accuracy', 'roc_auc']
logReg = LogisticRegression()
gridSearch = GridSearchCV(logReg, parameters, cv=5, scoring=scoring, refit='roc_auc') 

# "refit" gives the metric used deciding best model. 
# See more http://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html
gridSearch.fit(XTrain, yTrain.ravel())

def gridSearchSummary(method, scoring):
    """Prints best parameters from Grid search
    and AUC with standard deviation for all 
    parameter combos """
    
    method = eval(method)
    if scoring == 'accuracy':
        mean = 'mean_test_score'
        sd = 'std_test_score'
    elif scoring == 'auc':
        mean = 'mean_test_roc_auc'
        sd = 'std_test_roc_auc'
    print("Best: %f using %s" % (method.best_score_, method.best_params_))
    means = method.cv_results_[mean]
    stds = method.cv_results_[sd]
    params = method.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def createConfusionMatrix(method, printOut=True):
    """
    Computes and prints confusion matrices, accuracy scores,
    and AUC for test and training sets 
    """
    confusionArray = np.zeros(6, dtype=object)
    method = eval(method)
    
    # Train
    yPredTrain = method.predict(XTrain)
    yPredTrain = (yPredTrain > 0.5)
    cm = confusion_matrix(
        yTrain, yPredTrain) 
    cm = np.around(cm/cm.sum(axis=1)[:,None], 2)
    confusionArray[0] = cm
    
    accScore = accuracy_score(yTrain, yPredTrain)
    confusionArray[1] = accScore
    
    AUC = roc_auc_score(yTrain, yPredTrain)
    confusionArray[2] = AUC
    
    if printOut:
        print('\n###################  Training  ###############')
        print('\nTraining Confusion matrix: \n', cm)
        print('\nTraining Accuracy score: \n', accScore)
        print('\nTrain AUC: \n', AUC)
    
    # Test
    yPred = method.predict(XTest)
    yPred = (yPred > 0.5)
    cm = confusion_matrix(
        yTest, yPred) 
    cm = np.around(cm/cm.sum(axis=1)[:,None], 2)
    confusionArray[3] = cm
    
    accScore = accuracy_score(yTest, yPred)
    confusionArray[4] = accScore
    
    AUC = roc_auc_score(yTest, yPred)
    confusionArray[5] = AUC
    
    if printOut:
        print('\n###################  Testing  ###############')
        print('\nTest Confusion matrix: \n', cm)
        print('\nTest Accuracy score: \n', accScore)
        print('\nTestAUC: \n', AUC)    
    
    return confusionArray


import matplotlib.pyplot as plt
import seaborn
import scikitplot as skplt

seaborn.set(style="white", context="notebook", font_scale=1.5, 
            rc={"axes.grid": True, "legend.frameon": False,
"lines.markeredgewidth": 1.4, "lines.markersize": 10})
seaborn.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 4.5})

yPred = gridSearch.predict_proba(XTest) 
print(yTest.ravel().shape, yPred.shape)

#skplt.metrics.plot_cumulative_gain(yTest.ravel(), yPred_onehot)
skplt.metrics.plot_cumulative_gain(yTest.ravel(), yPred)

defaults = sum(yTest == 1)
total = len(yTest)
defaultRate = defaults/total
def bestCurve(defaults, total, defaultRate):
    x = np.linspace(0, 1, total)
    
    y1 = np.linspace(0, 1, defaults)
    y2 = np.ones(total-defaults)
    y3 = np.concatenate([y1,y2])
    return x, y3

x, best = bestCurve(defaults=defaults, total=total, defaultRate=defaultRate)    
plt.plot(x, best)    


plt.show()

#%%

"""The sigmoid function (or the logistic curve) is a
function that takes any real number, z, and outputs a number (0,1).
It is useful in neural networks for assigning weights on a relative scale.
The value z is the weighted sum of parameters involved in the learning algorithm."""

import matplotlib.pyplot as plt
import math as mt

z = numpy.arange(-5, 5, .1)
sigma_fn = numpy.vectorize(lambda z: np.exp(z)/(1+np.exp(z)))
sigma = sigma_fn(z)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z, sigma)
ax.set_ylim([-0.1, 1.1])
ax.set_xlim([-5,5])
ax.grid(True)
ax.set_xlabel('z')
ax.set_title('sigmoid function')

plt.show()

#%%
"""
Gradient descent example
"""
# Importing various packages
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys

# the number of datapoints
m = 30000
x = 2*np.random.rand(m,1)
y = 4+3*x+np.random.randn(m,1)

xb = np.c_[np.ones((m,1)), x]
beta_linreg = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y)
print(beta_linreg)

beta = np.random.randn(2,1)
eta = 0.1
Niterations = 1000

for iter in range(Niterations):
    gradients = 2.0/m*xb.T.dot(xb.dot(beta)-y)
    beta -= eta*gradients
   
print(beta)
xnew = np.array([[0],[2]])
xbnew = np.c_[np.ones((2,1)), xnew]
ypredict = xbnew.dot(beta)
ypredict2 = xbnew.dot(beta_linreg)
plt.figure()
plt.plot(xnew, ypredict, "b-")
plt.plot(xnew, ypredict2, "g-")
plt.plot(x, y ,'ro')
plt.axis([0,2.0,0, 15.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Gradient descent example')
plt.show()


#%%

"""
Defining functions
"""
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

#%%
'''
Finding Beta_logreg through gradient descent 
'''
#df=df.iloc[:,a:b] #plukker ut kol a til b av df

#Linear regression (ols) scikit
from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
y_pred_lin=linreg.fit(XTrain,yTrain.ravel()).predict(XTrain)#Prediction

#Logistic reg. scikit
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
y_pred_log=logreg.fit(XTrain,yTrain.ravel()).predict(XTrain) #X design matrix and y data
#pssibly ravel y_pred

print(mse(yTrain,y_pred_lin))
print(r2(yTrain,y_pred_lin))
#%%
plt.figure()
plt.hist(yTrain)
plt.show()

#%%
#Logistic regression own
beta=np.random.randn(26,1)
eta=0.001
n=np.size(yTrain)
cost = np.sum((XTrain.dot(beta) - yTrain)**2)/n #mse 
gradient=2*XTrain.T.dot(XTrain.dot(beta)-yTrain)
beta=beta-eta*gradient
y_prediction = XTrain.dot(beta) # X @ beta

#bytt Xtrain og ytrain til X og y

sigmoid=np.vectorize(lambda yTrain: np.exp(yTrain)/(1+np.exp(yTrain)))
sigma = sigmoid(yTrain)
plt.figure()
plt.plot(sigma)
plt.show()
#%%
plt.figure()
plt.plot(y_prediction)
plt.show()
#%%

# the number of datapoints
m = 30000#28497

xb = np.c_[np.ones((m,1)), x]
beta = np.random.randn(26,1)
y
eta = 0.1
Niterations = 1000

for iter in range(Niterations):
    gradients = 2.0/m*XTrain.T.dot(XTrain.dot(beta)-yTrain)#From slides. Why devide by m?
    beta -= eta*gradients#beta=beta-eta*gradients 

print(beta)

xnew = np.array([[0],[2]])
xnew = np.ones((30000,26))
#xbnew = np.c_[np.ones((2,1)), xnew]
xbnew = np.c_[np.ones((30000,0)), xnew]
ypredict = xbnew.dot(beta)
#ypredict2 = xbnew.dot(beta_linreg)
plt.figure()
plt.plot(xnew, ypredict, "b-")
#plt.plot(xnew, ypredict2, "g-")
plt.plot(x, y ,'ro')
#plt.axis([0,2.0,0, 15.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Gradient descent example')
plt.show()
#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import sklearn.neural_network
import sklearn.model_selection
import sklearn.metrics

np.random.seed(2019)


def franke(x, y):
    term = 3 / 4 * np.exp(-(9 * x - 2) ** 2 / 4 - (9 * y - 2) ** 2 / 4)
    term += 3 / 4 * np.exp(-(9 * x + 1) ** 2 / 49 - (9 * y + 1) / 10)
    term += 1 / 2 * np.exp(-(9 * x - 7) ** 2 / 4 - (9 * y - 3) ** 2 / 4)
    term -= 1 / 5 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)

    return term


L = 41

X, Y = np.meshgrid(np.linspace(0, 1, L), np.linspace(0, 1, L))
Z = franke(X, Y)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
#
# ax.plot_surface(X, Y, Z)
# ax.set_title("Franke's function")
#
# plt.show()

X_d = np.c_[X.ravel()[:, np.newaxis], Y.ravel()[:, np.newaxis]]
y_d = Z.ravel()[:, np.newaxis]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X_d, y_d, test_size=0.2
)


# Implement neural network
reg = sklearn.neural_network.MLPRegressor(
    hidden_layer_sizes=(100, 20),
    learning_rate="adaptive",
    learning_rate_init=0.01,
    max_iter=1000,
    tol=1e-7,
    verbose=True,
)
reg = reg.fit(X_train, y_train)

# See some statistics
pred = reg.predict(X_test)
print(f"MSE = {sklearn.metrics.mean_squared_error(y_test, pred)}")
print(f"R2 = {reg.score(X_test, y_test)}")

# Plot surface fit
pred = reg.predict(X_d)
Z_pred = pred.reshape(L, L)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
ax.plot_wireframe(X, Y, Z_pred)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(X, Y, np.abs(Z - Z_pred))
plt.show()


