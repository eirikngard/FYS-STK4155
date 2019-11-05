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
'''
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

df = df.drop(df[(df.PAY_0 == 0) |
                (df.PAY_2 == 0) |
                (df.PAY_3 == 0) |
                (df.PAY_4 == 0) |
                (df.PAY_5 == 0) |
                (df.PAY_6 == 0)].index)

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

'''
#Look at value count after clean up
for variable in variables:
    print('Variable '+variable)
    print('Value|Count')
    print(df[variable].value_counts())
    print('\n')
    
#Look again at correlation, after clean up

#print correlation
corr.loc['default payment next month']
'''
#%%
data=df.to_numpy()
X=data[:,:-1]
y=data[:,23]

#%%
'''
Defining designmatrix and data y
'''
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
 # Categorical variables to one-hot's
onehotencoder = OneHotEncoder(categories="auto")

X = ColumnTransformer(
    [("", onehotencoder, [1,2,3]),],
    remainder="passthrough"
).fit_transform(X)

#%%
'''
Splitting data in train and test data
'''

from sklearn.model_selection import train_test_split
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
Finding Beta through Gradient Descent (GD)
'''
#m=1896
beta = np.random.randn(29)*np.sqrt(1/29)
eta = 1e-5
Niterations = 1000

for iter in range(Niterations):
    p=np.exp(XTrain.dot(beta))/(1+np.exp(XTrain.dot(beta)))#sigoid
    gradients = XTrain.T.dot(p-yTrain) #gradients does not descend. Derivative of cost -X^T(y-p)
    beta = beta-eta*gradients
    
    #sjekk om cost minimeres 
y_predict_new = XTrain @ beta #y_predict_new = XTrain.dot(beta)

#print(f'MSE is {mse(yTrain,y_predict_new):.5f}')

#%%
'''
Calculate accuracy score for Logsistic regression
'''
#Own accuracy score for own logistic regression using own Gradient Descent 
y_pred=np.exp(y_predict_new)/(1+np.exp(y_predict_new))#prediksjon
y_pred[y_pred >= 0.5] = 1
y_pred[y_pred <= 0.5] = 0
Acc1=np.mean(y_pred == yTrain)

#Own accuracy score for logistic regression using scikit 
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(solver='lbfgs')
y_pred_log=logreg.fit(XTrain,yTrain.ravel()).predict(XTrain) #X design matrix and y data''
y_pred_log[y_pred_log >= 0.5] = 1
y_pred_log[y_pred_log <= 0.5] = 0
Acc2=np.mean(y_pred_log == yTrain)

#Accuracy from scikit on own logistic regression 
from sklearn.metrics import accuracy_score
Acc3=accuracy_score(yTrain, y_pred)#, normalize=False)

#%%
'''
Finding Beta through Stocastic (random) Gradient Descent (SGD)
'''
theta = np.random.randn(29)*np.sqrt(1/29)
eta = 1e-5

n_epochs = 50
t0, t1 = 5, 50
m=len(XTrain)
def learning_schedule(t):
    return t0/(t+t1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = XTrain[random_index:random_index+1]
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
#%%
#Using two variables beacause they use two in the article
print() 
print("Classification accuracy for LOGISTIC REGRESSION ")
print("--------------------------------")
print(f'Own accuracy, GD: {Acc1:.2f}') #Gir accuracy sore for prediksjonen 
print(f'Own accuracy, LOGREG by scikit: {Acc2:.2f}') #Gir accuracy sore for prediksjonen 
print(f'Scikit accuracy, GD: {Acc3:.2f}') #Gir accuracy sore for prediksjonen 
print(f'Own accuracy, SGD: {Acc_sgd:.2f}')
print()
print(f'MEAN: {(Acc_sgd+Acc1+Acc2+Acc3)/4:.2f}') 
print()
print("Article result:", 1-0.2)


#%%
"""""""""""""""""""""""""""""""""""""""""""""""""""
Feed Forward Neural Network code implementing
back propagation algorithm
"""""""""""""""""""""""""""""""""""""""""""""""""""
# Implement neural network
#This paragraph is from github (Øyvind or something)
import sklearn.neural_network
reg = sklearn.neural_network.MLPRegressor(
    hidden_layer_sizes=(100, 20),
    learning_rate="adaptive",
    learning_rate_init=0.01,
    max_iter=1000,
    tol=1e-7,
    verbose=True,
)
reg = reg.fit(XTrain, yTrain)

# See some statistics
pred = reg.predict(XTest)
print(f"MSE = {sklearn.metrics.mean_squared_error(yTest, pred)}")
print(f"R2 = {reg.score(XTest, yTest)}")
#%%
'''
# building our neural network

n_inputs, n_features = XTrain.shape
n_hidden_neurons = 50
n_categories = 10

# we make the weights normally distributed using numpy.random.randn

# weights and bias in the hidden layer
hidden_weights = np.random.randn(n_features, n_hidden_neurons)
hidden_bias = np.zeros(n_hidden_neurons) + 0.01

# weights and bias in the output layer
output_weights = np.random.randn(n_hidden_neurons, n_categories)
output_bias = np.zeros(n_categories) + 0.01
#%%
# setup the feed-forward pass, subscript h = hidden layer

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def feed_forward(X):
    # weighted sum of inputs to the hidden layer
    z_h = np.matmul(X, hidden_weights) + hidden_bias #X=a 
    # activation in the hidden layer
    a_h = sigmoid(z_h)
    
    # weighted sum of inputs to the output layer
    z_o = np.matmul(a_h, output_weights) + output_bias
    
    # softmax output
    # axis 0 holds each input and axis 1 the probabilities of each category
    exp_term = np.exp(z_o)
    probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
    
    return probabilities

probabilities = feed_forward(XTrain)


# we obtain a prediction by taking the class with the highest likelihood
def predict(X):
    probabilities = feed_forward(X)
    return np.argmax(probabilities, axis=1)

predictions = predict(XTrain)

#%%
# to categorical turns our integer vector into a onehot representation
from sklearn.metrics import accuracy_score

# one-hot in numpy
def to_categorical_numpy(integer_vector):
    integer_vector=yTrain
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 9# to get 10 entries in Y in backpropagation
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector

#onehottest = to_categorical_numpy(yTrain)#Not used for anything, just tested
#%%
#Y_train_onehot, Y_test_onehot = to_categorical(Y_train), to_categorical(Y_test)
Y_train_onehot, Y_test_onehot = to_categorical_numpy(yTrain), to_categorical_numpy(yTest)

def feed_forward_train(X):
    # weighted sum of inputs to the hidden layer
    z_h = np.matmul(X, hidden_weights) + hidden_bias
    # activation in the hidden layer
    a_h = sigmoid(z_h)
    
    # weighted sum of inputs to the output layer
    z_o = np.matmul(a_h, output_weights) + output_bias
    # softmax output
    # axis 0 holds each input and axis 1 the probabilities of each category
    exp_term = np.exp(z_o)
    probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
    
    # for backpropagation need activations in hidden and output layers
    return a_h, probabilities


#%%
def backpropagation(X, Y):
    a_h, probabilities = feed_forward_train(X)
    
    # error in the output layer
    error_output = probabilities - Y
    # error in the hidden layer
    error_hidden = np.matmul(error_output, output_weights.T) * a_h * (1 - a_h)
    
    # gradients for the output layer
    output_weights_gradient = np.matmul(a_h.T, error_output)
    output_bias_gradient = np.sum(error_output, axis=0)
    
    # gradient for the hidden layer
    hidden_weights_gradient = np.matmul(X.T, error_hidden)
    hidden_bias_gradient = np.sum(error_hidden, axis=0)

    return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient

#print("Old accuracy on training data: " + str(accuracy_score(predict(XTrain), yTrain)))

#%%
eta = 0.01
lmbd = 0.01
for i in range(1000):
    # calculate gradients
    dWo, dBo, dWh, dBh = backpropagation(XTrain, Y_train_onehot)
    
    # regularization term gradients
    dWo += lmbd * output_weights
    dWh += lmbd * hidden_weights
    
    # update weights and biases
    output_weights -= eta * dWo
    output_bias -= eta * dBo
    hidden_weights -= eta * dWh
    hidden_bias -= eta * dBh
    
print("Accuracy using NN with backpropagation: " + str(accuracy_score(predict(XTrain), yTrain)))
'''
#%%
'''
from NeuralNet_Nielsen import NeuralNetwork
lag = [1,2,3]
net = NeuralNetwork(lag)
train = np.vstack((XTrain, yTrain))
test = np.vstack((XTest, yTest))
net.SGD(XTrain,50,50,0.001,test_data=None)
#try something like this:
#3 hidden layers, 50 hidden neurons, 30 epochs, 500 batch size.
'''