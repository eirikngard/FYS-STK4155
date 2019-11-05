# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:25:54 2019

@author: Eirik N
"""

"""
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network. Gradients are calculated
using backpropagation. Note that I have focused on making the code
simple , easily readable , and easily modifiable. It is not optimized ,
and omits many desirable features.
"""
#### Libraries
# Standard library
import random
# Third -party libraries
import numpy as np

class NeuralNetwork:
    def __init__(self , layersize):
        """
        The list `layersize ` contains the number of neurons in the
        respective layers of the network. The biases and weights for the
        network are initialized randomly , using a Gaussian
        distribution with mean 0, and variance 1. Note that the first
        layer is assumed to be an input layer , and by convention we
        won't set any biases for those neurons , since biases are only
        ever used in computing the outputs from later layers.
        """
        self.num_layers = len(layersize)
        self.layersize = layersize
        self.biases = [np.random.randn(y, 1) for y in layersize[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layersize[:-1], layersize[1:])]

    def feedforward(self , a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases , self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def SGD(self , training_data , epochs , mini_batch_size , eta, test_data=None):
        """Train the neural network using mini -batch stochastic
        gradient descent. The ``training_data `` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs. The other non-optional parameters are
        self -explanatory. If ``test_data `` is provided then the
        network will be evaluated against the test data after each
        epoch , and partial progress printed out. This is useful for
        tracking progress , but slows things down substantially."""
        if test_data is not None:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                    training_data[k:k+mini_batch_size]
                    for k in range(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch , eta)
            print(mini_batches.shape)
        if test_data is not None:
            print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
        else:
            print ("Epoch {0} complete".format(j))
        #Remove this print statement later 

    def update_mini_batch(self , mini_batch , eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch `` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        new_b = [np.zeros(b.shape) for b in self.biases]
        new_w = [np.zeros(w.shape) for w in self.weights]
        print(mini_batch.shape)
        for x, y in mini_batch:
            delta_new_b , delta_new_w = self.backprop(x, y)
            new_b = [nb+dnb for nb, dnb in zip(new_b , delta_new_b)]
            new_w = [nw+dnw for nw, dnw in zip(new_w , delta_new_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights , new_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                        for b, nb in zip(self.biases , new_b)]
        
    def backprop(self , x, y):
        """Return a tuple ``(new_b , new_w)`` representing the
        gradient for the cost function C_x. ``new_b `` and
        ``new_w `` are layer -by-layer lists of numpy arrays , similar
        to ``self.biases `` and ``self.weights ``."""
        new_b = [np.zeros(b.shape) for b in self.biases]
        new_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations , layer by layer
        zs = [] # list to store all the z vectors , layer by layer
        for b, w in zip(self.biases , self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        new_b[-1] = delta
        new_w[-1] = np.dot(delta , activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book. Here ,
        # l = 1 means the last layer of neurons , l = 2 is the
        # second -last layer , and so on. It's a renumbering of the
        # scheme in the book , used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            new_b[-l] = delta
            new_w[-l] = np.dot(delta , activations[-l-1].transpose())
        return (new_b , new_w)
    
    def evaluate(self , test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
            for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def cost_derivative(self , output_activations , y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations -y)
    
    #### Miscellaneous functions
    def sigmoid(z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))
    
    def sigmoid_prime(z):
        """Derivative of the sigmoid function."""
        return sigmoid(z)*(1-sigmoid(z))

#%%
#Dette er en endring
import numpy as np

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

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

import sys
sys.path.append("class/")
from NeuralNetwork import NeuralNetwork
import projectfunctions as pf

sns.set()
sns.set_style("whitegrid")
sns.set_palette("husl")

filepath = "../data/input/"
filename = "default_of_credit_card_clients"

df = pd.read_pickle(filepath + filename + "_partial_clean.pkl")
#print(df.head())

"""
data = df.to_numpy()
labels = data[:, -1]
input = data[:, :-1]
sc = StandardScaler()
input = sc.fit_transform(input)
"""

# preparing designmatrix by scaling and using one hot encoding for cat data
input = df.loc[:, df.columns != 'default payment next month']
column_indices = pf.pca(input)
print(column_indices)
input = input.iloc[:, column_indices]
num_attributes = list(input.drop(["SEX", "EDUCATION", "MARRIAGE"], axis=1))
cat_attributes = list(input.iloc[:, 1:4])
#num_attributes = list(input.drop(["SEX", "EDUCATION", "MARRIAGE",'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'], axis=1))
#cat_attributes = list(input.iloc[:, 1:4]) + list(input.iloc[:,5:11])

input_pipeline = ColumnTransformer([
                                    ("scaler", StandardScaler(), num_attributes),
                                    ("onehot", OneHotEncoder(categories="auto"), cat_attributes)
                                    ],
                                    remainder="passthrough"
                                    )
input_prepared = input_pipeline.fit_transform(input)

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
