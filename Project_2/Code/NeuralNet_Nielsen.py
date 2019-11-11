# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:25:54 2019

@author: Eirik Nordgård
"""

"""
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network. Gradients are calculated
using backpropagation. 

NOTE: I was not able to finnish the neural network code. This is a start, 
with basis in Nielsens code on neural network for hand-written digits. 
"""

import random
import numpy as np

def layersize:
    """
    The list `layersize ` contains the number of neurons in the
    respective layers of the network. The biases and weights for the
    network are initialized randomly. Note that the first
    layer is assumed to be an input layer.
    """
    num_layers = len(layersize)
    layersize = layersize
    biases = [np.random.randn(y, 1) for y in layersize[1:]]
    weights = [np.random.randn(y, x) for x, y in zip(layersize[:-1], layersize[1:])]

def feedforward(a):
    """Return the output of the network if ``a`` is input."""
    for b, w in zip(biases , weights):
        a = sigmoid(np.dot(w, a)+b)
    return a

def SGD(training_data , epochs , mini_batch_size , eta, test_data=None):
    """Train the neural network using mini -batch stochastic
    gradient descent. The ``training_data `` is a list of tuples
    ``(x, y)`` representing the training inputs and the desired
    outputs. If ``test_data `` is provided then the
    network will be evaluated against the test data after each
    epoch , and partial progress printed out."""
    if test_data is not None:
        n_test = len(test_data)
    n = len(training_data)
    for j in range(epochs):
        random.shuffle(training_data)
        mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
    for mini_batch in mini_batches:
        update_mini_batch(mini_batch , eta)
        print(mini_batches.shape)
    if test_data is not None:
        print ("Epoch {0}: {1} / {2}".format(j, evaluate(test_data), n_test))
    else:
        print ("Epoch {0} complete".format(j))
    #Remove this print statement later 
    
def backprop(x, y):
    """Return a tuple ``(new_b , new_w)`` representing the
    gradient for the cost function C_x. ``new_b `` and
    ``new_w `` are layer -by-layer lists of numpy arrays , similar
    to ``biases `` and ``weights ``."""
    new_b = [np.zeros(b.shape) for b in biases]
    new_w = [np.zeros(w.shape) for w in weights]
    # feedforward
    activation = x
    activations = [x] # list to store all the activations , layer by layer
    zs = [] # list to store all the z vectors , layer by layer
    for b, w in zip(biases , weights):
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)
    # backward pass
    delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
    new_b[-1] = delta
    new_w[-1] = np.dot(delta , activations[-2].transpose())
    
    for l in range(2, num_layers):
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(weights[-l+1].transpose(), delta) * sp
        new_b[-l] = delta
        new_w[-l] = np.dot(delta , activations[-l-1].transpose())
    return (new_b , new_w)

def evaluate(test_data):
    """Return the number of test inputs for which the neural
    network outputs the correct result. Note that the neural
    network's output is assumed to be the index of whichever
    neuron in the final layer has the highest activation."""
    test_results = [(np.argmax(feedforward(x)), y)
        for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)

def cost_derivative(output_activations , y):
    """Return the vector of partial derivatives \partial C_x /
    \partial a for the output activations."""
    return (output_activations -y)

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

#%%

