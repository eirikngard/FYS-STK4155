# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 08:14:38 2019

@author: Eirik N
"""

class NeuralNetwork:
    def __init__(self, layer_sizes):

        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)

        #initialize weights and biases with random numbers
        self.biases = [np.random.randn(size) for size in layer_sizes[1:]]
        self.weights = [np.random.randn(size, size_prew) for size_prew, size \
                        in zip(layer_sizes[:-1], layer_sizes[1:])]

    def feedforward(self, input):
        """
        Function for feeding the input through the Network
            input = array with inputs to the first layer of the network
            returns array with resulting output from the last layer
        """
        for layer in range(self.n_layers-1):
            bias2D = self.biases[layer][np.newaxis]
            z = np.matmul(input,self.weights[layer].transpose()) + bias2D
            input = self.sigmoid(z)
        return input.transpose()[0]

    def backpropagation(self, input, labels):
        """
        Function for calculationg the backwards propagating correction of the
        weights and biases, given a learning rate, using gradient descent
        """
        biases_gradient = [np.zeros(bias.shape) for bias in self.biases]
        weights_gradient = [np.zeros(weight.shape) for weight in self.weights]
        activation = input
        activations = [activation]
        zs = []

        for layer in range(self.n_layers-1):
            z = np.matmul(self.weights[layer], activation) + self.biases[layer]
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1],labels)*self.sigmoid_derivative(zs[-1])
        biases_gradient[-1] = delta
        #add new axis so that python handles matrix multiplication
        activation2D = activations[-2][np.newaxis]
        weights_gradient[-1] = np.matmul(delta, activation2D)

        for layer in range(2, self.n_layers):
            z = zs[-layer]
            delta = np.dot(self.weights[-layer+1].transpose(), delta)*self.sigmoid_derivative(z)
            biases_gradient[-layer] = delta
            #add new axis so that python handles matrix multiplication
            activation2D = activations[-layer-1][np.newaxis]
            delta2D = delta[np.newaxis].transpose()
            weights_gradient[-layer] = np.matmul(delta2D, activation2D)
        return biases_gradient, weights_gradient


    def train(self, training_input, training_labels ,n_epochs, batch_size, \
              learning_rate, test_input=None, test_labels=None, test=False):
        #kode for stochastic gradient decent
        n = len(training_labels)
        for epoch in range(n_epochs):
            idx = np.arange(n)
            np.random.shuffle(idx)
            training_input = training_input[idx]
            training_labels = training_labels[idx]
            labels_mini_batches = [training_labels[i:i+batch_size] for i in range(0, n, batch_size)]
            input_mini_batches = [training_input[i:i+batch_size] for i in range(0, n, batch_size)]
            for labels_mini_batch, input_mini_batch in zip(labels_mini_batches, input_mini_batches):
                biases_gradient = [np.zeros(bias.shape) for bias in self.biases]
                weights_gradient = [np.zeros(weight.shape) for weight in self.weights]
                for label, input in zip(labels_mini_batch, input_mini_batch):
                    delta_bias_gradient, delta_weight_gradient= self.backpropagation(input, label)
                    biases_gradient = [bg + dbg for  bg, dbg in zip(biases_gradient, delta_bias_gradient)]
                    weights_gradient = [wg + dwg for  wg, dwg in zip(weights_gradient, delta_weight_gradient)]
                self.biases = [b - learning_rate*bg for b, bg in zip(self.biases, biases_gradient)]
                self.weights = [w - learning_rate*wg for w, wg in zip(self.weights, weights_gradient)]

            if test:
                print('Epoch {}: {:.3f} correct'.format(epoch, self.evaluate(test_input, test_labels)))
            else:
                print('Epoch {} complete'.format(epoch))

    def predict(self, input):
        """
        Function for applying the network on (new) input.
            input = array of inputs to the first layer
        Returns arrays with predictions
        """
        probabilities = self.feedforward(input)
        probabilities_array = np.empty(len(probabilities),dtype=np.uint)
        for i in range(len(probabilities)):
            if probabilities[i] > 0.5:
                probabilities_array[i] = 1
            if probabilities[i] <= 0.5:
                probabilities_array[i] = 0
        return probabilities_array

    def evaluate(self, input, labels):
        predictions = self.predict(input)
        count = 0
        for prediction, target in zip(predictions, labels):
            if prediction == target:
                count += 1
        return count/len(labels)

    def predict_probabilities(self, input):
        """
        Function for applying the network on (new) input.
            input = array of inputs to the first layer
        Returns the probability output
        """
        probabilities = self.feedforward(input)
        return probabilities

    def cost_derivative(self, output_activations, labels):
        return output_activations-labels

    def sigmoid(self, z):
        return np.exp(z)/(1+np.exp(z))

    def sigmoid_derivative(self, z):
        return np.exp(z)/(1 + np.exp(z))**2
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

import sys
sys.path.append("class/")

#%%
#Importing data

sns.set_style('darkgrid')
np.random.seed(98)
cwd = os.getcwd()
filename = cwd + '/default of credit card clients.xls'
#load data set
df = pd.read_excel(filename, index_col=0,skiprows=[0])
#show first 5 rows
df.head()

#%%
#Clean-up in data 

print('# of entries before clean up: {}'.format(len(df.index)))

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
#%%
# preparing designmatrix by scaling and using one hot encoding for cat data
input = df.loc[:, df.columns != 'default payment next month']

#Numerical and categorical attributes
num_attributes = list(input.drop(["SEX", "EDUCATION", "MARRIAGE"], axis=1))
cat_attributes = list(input.iloc[:, 1:10]) #includes PAY_0-6
#%%


# Input Scaling
#sc = StandardScaler()
#XTrain[:,-14:] = sc.fit_transform(XTrain[:,-14:])
#XTest[:,-14:] = sc.transform(XTest[:,-14:])
#%%

input_data_raw = ColumnTransformer([
                                    ("scaler", StandardScaler(), num_attributes),
                                    ("onehot", OneHotEncoder(categories="auto"), cat_attributes)
                                    ],
                                    remainder="passthrough"
                                    )
input_prepared = input_data_raw.fit_transform(input)
#%%

# exporting labels to a numpy array
labels = df.loc[:, df.columns == 'default payment next month'].to_numpy().ravel()

first_layer = input_prepared.shape[1]
layers = [first_layer, 20, 20, 1]
n_epochs = 5
batch_size = 100
learning_rate = 0.1

trainingShare = 0.8
seed  = 42
training_input, test_input, training_labels, test_labels = train_test_split(
                                                                input_prepared,
                                                                labels,
                                                                train_size=trainingShare,
                                                                test_size = 1-trainingShare,
                                                                random_state=seed
                                                                )

network = NeuralNetwork(layers)
network.train(training_input, training_labels ,n_epochs, batch_size, \
                learning_rate, test_input, test_labels, test=True)

output = network.predict_probabilities(test_input)

plt.hist(output)
plt.xlim([0,1])
plt.title('Histogram of output from NN')
plt.xlabel('output')
plt.ylabel('count')
plt.show()
#%%
import seaborn as sns

#fig, ax = plt.subplots(figsize = (10, 10))
plt.figure()
sns.heatmap(output)#, annot=True, ax=ax, cmap="viridis")
plt.title("Training Accuracy")
plt.ylabel("$\eta$ logspace")
plt.xlabel("epoch")
plt.show()
#%%
"""
etas = np.logspace(-6,0,7)
epo=np.arange(20)

train_accuracy = np.zeros((len(etas),len(epo)))
test_accuracy = np.zeros((len(etas),len(epo)))
for i,e in enumerate(etas):
    for ep in range(20): 
        w =  sto_grad_des(XTrain,yTrain,epochs=ep,batch_size=20,eta2=e)
        
        test_result = predict(XTest,w)
        train_result = predict(XTrain,w)
        
        test_acc = accuracy(test_result,yTest) 
        train_acc = accuracy(train_result,yTrain)
        
        #test_sci_acc = accuracy_score(yTest, test_result)
        #train_sci_acc = accuracy_score(yTrain, train_result)
        
        train_accuracy[i,ep] = accuracy_score(yTrain,train_result)
        test_accuracy[i,ep] = accuracy_score(yTest,test_result)
"""