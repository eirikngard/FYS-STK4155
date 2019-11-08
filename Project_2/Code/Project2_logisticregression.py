# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:40:58 2019

@author: Eirik Nordgård
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
#matplotlib inline
sns.set_style('darkgrid')
np.random.seed(98)

cwd = os.getcwd()
filename = cwd + '/default of credit card clients.xls'

#load data set
df = pd.read_excel(filename, index_col=0,skiprows=[0])

#show first 5 rows
df.head()

#%%

print('# of entries before clean up: {}'.format(len(df.index)))

# Remove instances with zeros only for past bill statements and paid amounts

# Remove illegal education value
df = df.drop(df[(df.EDUCATION == 0) |
                (df.EDUCATION == 5) |
                (df.EDUCATION == 6)].index)

# Remove illegal marriage value
df = df.drop(df[(df.MARRIAGE == 0)].index)

# Remove illegal pay value
df = df.drop(df[(df.PAY_0 == -2) |
                (df.PAY_2 == -2) |
                (df.PAY_3 == -2) |
                (df.PAY_4 == -2) |
                (df.PAY_5 == -2) |
                (df.PAY_6 == -2)].index)
"""
df = df.drop(df[(df.PAY_0 == 0) |
                (df.PAY_2 == 0) |
                (df.PAY_3 == 0) |
                (df.PAY_4 == 0) |
                (df.PAY_5 == 0) |
                (df.PAY_6 == 0)].index)
"""
# Remove negative bill and pay amounts
df = df.drop(df[(df.BILL_AMT1 < 0) |
                (df.BILL_AMT2 < 0) |
                (df.BILL_AMT3 < 0) |
                (df.BILL_AMT4 < 0) |
                (df.BILL_AMT5 < 0) |
                (df.BILL_AMT6 < 0)].index)

df = df.drop(df[(df.PAY_AMT1 < 0) |
                (df.PAY_AMT2 < 0) |
                (df.PAY_AMT3 < 0) |
                (df.PAY_AMT4 < 0) |
                (df.PAY_AMT5 < 0) |
                (df.PAY_AMT6 < 0)].index)

print('# of entries after clean up: {}'.format(len(df.index)))


data=df.to_numpy()
X=data[:,:-1]
y=data[:,23] #Output

#%%
'''
Defining designmatrix and data y
'''
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
 # Categorical variables to one-hot's
onehotencoder = OneHotEncoder(categories="auto")

#OneHotEncode all categorical feature (except age/column 4)
X = ColumnTransformer(
    [("", onehotencoder, [1,2,3,5,6,7,8,9,10]),],
    remainder="passthrough",sparse_threshold=0
).fit_transform(X)

#%%
'''
Splitting data in train and test data
'''

from sklearn.model_selection import train_test_split
# Train-test split, using trainingshare of 0.8
trainingShare = 0.8 
seed  = 1
XTrain, XTest, yTrain, yTest = train_test_split(X, y, train_size=trainingShare, \
                                              test_size = 1-trainingShare,
                                              random_state=seed)

# Input Scaling of the remaining columns (the last 14).
sc = StandardScaler(with_mean=False)
XTrain[:,-14:] = sc.fit_transform(XTrain[:,-14:])
XTest[:,-14:] = sc.transform(XTest[:,-14:])

#%%
#Downsampling, correcting for scewed distribution
all_=np.where(yTrain==1)
some_=np.where(yTrain==0)
some_[0][:len(all_[0])]
sample_idx = np.concatenate((all_[0], some_[0][:len(all_[0])]))

XTrain = XTrain[sample_idx]
yTrain=yTrain[sample_idx]

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


#(Normal) Gradient Descent (GD)

def normal_gradient_descent(eta=1e-5,Niterations=1000):
    #m=1896
    beta = np.random.randn(X.shape[1])*np.sqrt(1/X.shape[1])
    #eta = 1e-5
    #Niterations = 1000
    
    for iter in range(Niterations):
        p=np.exp(XTrain.dot(beta))/(1+np.exp(XTrain.dot(beta)))#sigoid
        gradients = XTrain.T.dot(p-yTrain) #gradients does not descend. Derivative of cost -X^T(y-p)
        beta = beta-eta*gradients
    return beta

#Stochastic Gradient Descent (SGD)
def sto_grad_des(X,Y,epochs=40,batch_size=200,eta2=1e-4):
    #theta2 = np.random.randn(75)*np.sqrt(1/75)
    theta2 = np.random.randn(X.shape[1])*np.sqrt(1/X.shape[1])
   
    n_samples, n_cols = X.shape
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    #batch_size=100
    splits = int(n_samples/batch_size)
    batches = np.array_split(idx,splits)
    
    for i in range(epochs):
        np.random.shuffle(X) 
        for b_idx in batches:
            #b_idx = np.random.randint(m)
            xi2 = X[b_idx]
            yi2 = Y[b_idx]
            #print(yi2)
            p12=np.exp(xi2.dot(theta2))/(1+np.exp(xi2.dot(theta2)))#sigmoid
            #print(p12)
            gradient2 = xi2.T.dot(p12-yi2)/batch_size
            #print(gradient2)
            theta2 = theta2 - eta2*gradient2
            
    return theta2

def predict(X,theta):
    pred = X @ theta
    
    y_pred = np.exp(pred)/(1+np.exp(pred))#prediksjon, y gjennom sigmoid
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    return y_pred
    #return accuracy*100

def accuracy(prediction,Y):
    accuracy = np.mean(prediction == Y)
    return accuracy*100


#%%
'''
Testing GD
'''
beta=normal_gradient_descent()
y_predict_new = predict(XTrain,beta) 
print("Acuracy score GD: {:.3f}".format(accuracy(y_predict_new,yTrain)))
print(f'MSE is {mse(yTrain,y_predict_new):.3f}')

#%%
'''
Calculate accuracy score for Logsistic regression
'''
#Own accuracy score for logistic regression using scikit 
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(solver='lbfgs',fit_intercept=False,penalty='l2')
y_pred_log=logreg.fit(XTrain,yTrain.ravel()).predict(XTrain) #X design matrix and y data''
accuracy_logreg_sci = accuracy(y_pred_log,yTrain)
print("Accuracy for logistic legression by scikit: {:.3f}".format(accuracy_logreg_sci))

#Accuracy from scikit on own logistic regression 
from sklearn.metrics import accuracy_score
acc_by_sci_logreg = accuracy_score(yTrain, y_pred_log)#, normalize=False)
print("Accuracy bu scikit for logistic legression by scikit: {:.3f} \n"\
      .format(acc_by_sci_logreg))

#%%
'''
Finding Beta through Stocastic (random) Gradient Descent (SGD)
'''
"""
theta = np.random.randn(75)*np.sqrt(1/75)
eta = 1e-5

n_epochs = 50
t0, t1 = 5, 50
m=len(XTrain)
def learning_schedule(t):
    return t0/(t+t1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = XTrain[random_index:random_index+1]#b_idx
        yi = yTrain[random_index:random_index+1]
        p1=np.exp(xi.dot(theta))/(1+np.exp(xi.dot(theta)))#sigoid
        #gradient = xi.T @ ((xi @ theta)-yi)
        gradient = xi.T.dot(p1-yi)
        eta = learning_schedule(epoch*m+i)
        theta = theta - eta*gradient

#PROBLEM: gives a littlebit different accuracy each run of this section
#PROBLEM: need to find the what epocs and batches that ptovides best accuracy

pred_sgd = XTrain @ theta

y_pred_sgd=np.exp(pred_sgd)/(1+np.exp(pred_sgd))#prediksjon, y gjennom sigmoid
y_pred_sgd[y_pred_sgd >= 0.5] = 1
y_pred_sgd[y_pred_sgd <= 0.5] = 0

Acc_sgd=np.mean(y_pred_sgd == yTrain)
"""
#%%
'''
Finding Beta through Stocastic (random) Gradient Descent (SGD) VERSION 2
Med denne trenger du ikke mini_batch_update i NN
'''
from sklearn.linear_model import SGDClassifier
etas = np.logspace(-6,0,7)
epo=np.arange(20)
train_accuracy = np.zeros((len(etas),len(epo)))
test_accuracy = np.zeros((len(etas),len(epo)))
for i,e in enumerate(etas):
    for ep in range(20): 
        w =  sto_grad_des(XTrain,yTrain,epochs=20,batch_size=100,eta2=1e-3)
        
        test_result = predict(XTest,w)
        train_result = predict(XTrain,w)
        
        test_acc = accuracy(test_result,yTest) 
        train_acc = accuracy(train_result,yTrain)
        
        #test_sci_acc = accuracy_score(yTest, test_result)
        #train_sci_acc = accuracy_score(yTrain, train_result)
        
        train_accuracy[i,ep] = accuracy_score(yTrain,train_result)
        test_accuracy[i,ep] = accuracy_score(yTest,test_result)
        
        #sjekk om prediskjon i sklearn matcher med min. Kan være at jeg har for
        #få epoker, eller at datasettet ikke lar seg løse med logistic regression
        #Stemmer heatmap fra scikit med mitt. Hvis det gjør det, så er 
        #det greit at det ser radom ut
#%%
"""
SGD by scikit-learn
"""
clf = SGDClassifier()
y_pred_scikit = clf.fit(XTrain,yTrain.ravel()).predict(XTrain)
train_accuracy_sci = accuracy_score(yTrain,y_pred_scikit)*100 
train_accuracy_own = accuracy(y_pred_scikit,yTrain)

print("Acc by me on SGD by scikit: {:.3f}".format(train_accuracy_own))
print("Acc by scikit on SGD by scikit: {:.3f}".format(train_accuracy_sci))
#%% 
"""
Visualisering with a heatmap
"""
import seaborn as sns

plt.figure()
sns.heatmap(train_accuracy)#, annot=True, ax=ax, cmap="viridis")
plt.title("Training Accuracy")
plt.ylabel("$\eta$ logspaced learning parameter")
plt.xlabel("epoch")
plt.show()

plt.figure()
sns.heatmap(test_accuracy)#, annot=True, ax=ax, cmap="viridis")
#plt.colorbar()
plt.title("Test Accuracy")
plt.ylabel("$\eta$ logspaced learning parameter")
plt.xlabel("epoch")
plt.show()
#%%
#Trying a subplot with the two figures above on same color scale
fig, (ax, ax2, cax) = plt.subplots(ncols=3,figsize=(8,8),
      gridspec_kw={"width_ratios":[1,1,0.1]})
plt.suptitle("Heatmap of training and test accuracy with varying epoch and eta")
fig.subplots_adjust(wspace=0.1)
im=ax.imshow(train_accuracy,vmin=0.2,vmax=0.9)
im2=ax2.imshow(test_accuracy,vmin=0.2,vmax=0.9)
fig.colorbar(im,cax=cax)
plt.show

#%%

#%%
#Using two variables beacause they use two in the article

"""
Feed Forward Neural Network code implementing
back propagation algorithm
"""
# Implement neural network
#This paragraph is from github (Øyvind or something)
import sklearn.neural_network
network = sklearn.neural_network.MLPRegressor(
    hidden_layer_sizes=(20, 20),activation='logistic',
    learning_rate="adaptive",
    learning_rate_init=0.01,batch_size=2000,
    max_iter=200,
    tol=1e-4,
    verbose=True,
)
#reg = reg.fit(XTrain, yTrain)
reg = network.fit(XTrain, yTrain)
pred=reg.predict(XTest)

pred[pred >= 0.5] = 1
pred[pred < 0.5] = 0

real_accuracy_nn = accuracy(pred,yTest)
print("Accuracy NN by scikit: {:.3f}".format(real_accuracy_nn))

#Kan skrive i rapport at for videre arbeid, prøv med/uten downsampling. 

#All over, my accuracy scores is way better if not removing all the 
#0s in PAY data, but if i keep them, my SGD fails, returning only nan.

# For future work: look at other parameters to evaluate the network
# and regression methods.
#%%

#%%
"""
from NeuralNet_Nielsen import NeuralNetwork
lag = [1,2,3]
net = NeuralNetwork(lag)
train = np.vstack((XTrain.T, yTrain))
#test = np.vstack((XTest, yTest))
#SGD(self , training_data , epochs , mini_batch_size , eta, test_data=None):
#net.SGD(training_data , 30, 10, 3.0, test_data=test_data)
k=(2,2)
dat=(XTrain,yTrain)
net.SGD(train,30,10,0.01,test_data=None)
#try something like this:
#3 hidden layers, 50 hidden neurons, 30 epochs, 500 batch size.
"""
#%%

print() 
print("Classification accuracy for LOGISTIC REGRESSION ")
print("--------------------------------")
print("Acuracy score GD: {:.3f}".format(accuracy(y_predict_new,yTrain)))
print("Accuracy for logistic legression by scikit: {:.3f}".format(accuracy_logreg_sci))
print("Accuracy bu scikit for logistic legression by scikit: {:.3f} \n"\
      .format(acc_by_sci_logreg))
print("Acc by me on SGD by scikit: {:.3f}".format(train_accuracy_own))
print("Acc by scikit on SGD by scikit: {:.3f}".format(train_accuracy_sci))
print("Accuracy NN by scikit: {:.3f}".format(real_accuracy_nn))


#print(f'MEAN: {(Acc_sgd+Acc1+Acc2+Acc3)/4:.2f}') 
print()
print("Article result:", 1-0.2)

