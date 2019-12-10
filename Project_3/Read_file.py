# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 08:38:30 2019

@author: Eirik Nordgård 
"""
# Load the Pandas libraries with alias 'pd' 
import pandas as pd 

df = pd.read_csv("seattleWeather_1948-2017.csv")
# Preview the first 5 lines of the loaded data 
df.head()    

import matplotlib.pyplot as plt

#Convert inces and farengeit to mm and celcius
#°F = 9/5*°C + 32 
#°C = 5/9*(°F - 32)
# 1 inch = 25,4 mm
data = df.to_numpy()
day = data[:,0]
prcp_c = [i * 24.5 for i in data[:,1]]
tmax_c = [(i - 32)*(5/9) for i in data[:,2]]
tmin_c = [(i - 32)*(5/9) for i in data[:,3]]
rain = data[:,4]


#%%

fig, ax = plt.subplots(3, sharex = True)
fig.suptitle('Seattle Weather Data 1948-2017')
ax[0].plot(prcp_c)
ax[0].set_title('Precipitation [mm]')
ax[1].plot(tmin_c,'g')
ax[1].set_title('TMIN [\N{DEGREE SIGN}C]')
ax[2].plot(tmax_c,'r')
ax[2].set_title('TMAX [\N{DEGREE SIGN}C]')


#Adjusting the labeling of the figure 
#for ax in axs.flat:
#    ax.set(xlabel='x-label', ylabel='y-label')
# Hide x labels and tick labels for top plots and y ticks for right plots.
#for ax in axs.flat:
#    ax.label_outer()
#%%
import collections, numpy as np

#Count false/true in rain
#Removing nan values. Remove values in the corresponding indexes as well 
rain = rain[~pd.isnull(rain)] #removing 'nan'
collections.Counter(rain)

#Convert true/false to 1/0
rain_bin = rain.astype(int) #must remove nan before I do this. False=0. True=1
#%%
from sklearn.model_selection import train_test_split
# Train-test split
trainingShare = 0.8 
seed  = 1
x_des = np.asarray([prcp_c,tmin_c,tmax_c]).T
x_des = x_des[:-3,:] #Have removed the last three, REMEBER TO REMOVE THE CORRESPONDING TO THE NANs
xtrain, xtest, ytrain, ytest = train_test_split(x_des, rain_bin, train_size=trainingShare, \
                                              test_size = 1-trainingShare,
                                              random_state=seed)
#%%
# Scaling the input in a way?
#%%
def sto_grad_des(X,Y,epochs=40,batch_size=200,eta2=1e-4):
    #theta2 = np.random.randn(75)*np.sqrt(1/75)
    beta = np.random.randn(X.shape[1])*np.sqrt(1/X.shape[1])
   
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
            xi = X[b_idx]
            yi = Y[b_idx]
            #print(yi2)
            p2=np.exp(xi.dot(beta))/(1+np.exp(xi.dot(beta)))#sigmoid
            #print(p12)
            gradient = xi.T.dot(p2-yi)/batch_size
            #print(gradient2)
            beta = beta - eta2*gradient
            
    return beta

def predict(X,beta):
    pred = X @ beta
    
    y_pred = np.exp(pred)/(1+np.exp(pred))#prediksjon, y gjennom sigmoid
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    return y_pred
    #return accuracy*100

def accuracy(prediction,Y):
    accuracy = np.mean(prediction == Y)
    return accuracy*100
#%%
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
etas = np.logspace(-6,0,7)
epo=np.arange(20)
train_accuracy = np.zeros((len(etas),len(epo)))
test_accuracy = np.zeros((len(etas),len(epo)))
for i,e in enumerate(etas):
    for ep in range(20): 
        w =  sto_grad_des(xtrain,ytrain,epochs=20,batch_size=100,eta2=1e-3)
        
        test_result = predict(xtest,w)
        train_result = predict(xtrain,w)
        
        test_acc = accuracy(test_result,ytest) 
        train_acc = accuracy(train_result,ytrain)
        
        #test_sci_acc = accuracy_score(yTest, test_result)
        #train_sci_acc = accuracy_score(yTrain, train_result)
        
        train_accuracy[i,ep] = accuracy_score(ytrain,train_result)
        test_accuracy[i,ep] = accuracy_score(ytest,test_result)
             
        #sjekk om prediskjon i sklearn matcher med min. Kan være at jeg har for
        #få epoker, eller at datasettet ikke lar seg løse med logistic regression
        #Stemmer heatmap fra scikit med mitt. Hvis det gjør det, så er 
        #det greit at det ser radom ut
print("Accuracy SGD: {}".format(np.mean(train_accuracy)))


