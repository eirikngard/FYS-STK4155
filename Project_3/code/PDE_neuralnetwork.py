# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 09:45:18 2019

@author: Eirik Nordgård
"""
"""
I left this code in git because is produces 3D plots of the neuralnet solution,
which does not work in the script "Network" as it should. 
"""

import numpy as np 
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import tensorflow.compat.v1 as tf


def neural_network(num_iter=1000):
    tf.disable_v2_behavior()

    # Reset the graph 
    tf.reset_default_graph()

    # Setting a seed  
    tf.set_random_seed(4155)
    # Construct each possible point pair (x,t) to feed the neural network
    Nx = 10; Nt = 10
    x = np.linspace(0, 1, Nx) #from 0 to 1 (sin function)
    t = np.linspace(0,1,Nt)
    
    X,T = np.meshgrid(x, t)
    
    x_ = (X.ravel()).reshape(-1,1)
    t_ = (T.ravel()).reshape(-1,1)
    
    x_tf = tf.convert_to_tensor(x_)
    t_tf = tf.convert_to_tensor(t_) #converts x and t to tensors
    
    points = tf.concat([x_tf,t_tf],1)#concatenates to one dimention
    

    
    num_hidden_neurons = [20,20]
    num_hidden_layers = np.size(num_hidden_neurons)
    
    with tf.variable_scope('nn'): #DeepNeuralNetwork
        # Input layer
        previous_layer = points
        
        # Hidden layers
        for l in range(num_hidden_layers):
            current_layer = tf.layers.dense(previous_layer, \
                                            num_hidden_neurons[l],\
                                            activation=tf.nn.sigmoid)
            previous_layer = current_layer
             
        
        # Output layer
        nn_output = tf.layers.dense(previous_layer, 1)
        #Dense implements the operation: 
        #output = activation(dot(input, kernel) + bias)
    
        
    # Set up the cost function
    def u(x):
        return tf.sin(np.pi*x) #This is initial condition
    
    #Trial solution
    with tf.name_scope('cost'):
        trial = (1-t_tf)*u(x_tf) + x_tf*(1-x_tf)*t_tf*nn_output
    
        trial_dt = tf.gradients(trial,t_tf)
        trial_d2x = tf.gradients(tf.gradients(trial,x_tf),x_tf)
    
        err = tf.square(trial_dt[0] - trial_d2x[0])
        cost = tf.reduce_sum(err, name = 'cost')
    
    # Define how the neural network should be trained
    learning_rate = 0.001
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        traning_op = optimizer.minimize(cost)
        #Adam is an optimization algorithm that can be used instead of the classical 
        #stochastic gradient descent procedure to update network weights iterative 
        #based in training data. Could also use GradientDescentOptimizer
    
    # Reference variable to the output from the network
    u_nn = None
    
    # Define a node that initializes all the nodes within the computational graph
    # for TensorFlow to evaluate
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess: #Class for tunning tensorflow operations
        # Initialize the computational graph
        init.run()
        
        #print('Initial cost: %g'%cost.eval())
        
        for i in range(num_iter):
            sess.run(traning_op)
    
        #print('Final cost: %g'%cost.eval())
        
        u_nn = trial.eval()
    
    u_e = np.exp(-np.pi**2*t_)*np.sin(np.pi*x_) #exact/analytical solution
    
    U_nn = u_nn.reshape((Nt,Nx)).T #de første og den siste er 0. Hva betyr det?
    U_e = u_e.reshape((Nt,Nx)).T

    return U_nn, U_e, x
#%%

Nx = 100; Nt = 10
x = np.linspace(0, 1, Nx) #from 0 to 1 (sin function)
t = np.linspace(0,1,Nt)

U_nn,U_e,x = neural_network(1000)
diff_mat = np.abs(U_e - U_nn)

figdir = "../figures/"
# Surface plot of the solutions

T,X = np.meshgrid(t, x)

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title("Neural Network solution",fontsize=35)
s = ax.plot_surface(T,X,U_nn,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_xlabel('Time $t$',fontsize=35)
ax.set_ylabel('Position $x$',fontsize=35);
#fig.savefig(figdir + "dnn.png")

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Exact solution',fontsize=35)
s = ax.plot_surface(T,X,U_e,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_xlabel('Time $t$',fontsize=35)
ax.set_ylabel('Position $x$',fontsize=35);
#fig.savefig(figdir + "exact.png")

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Difference',fontsize=35)
s = ax.plot_surface(T,X,diff_mat,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_xlabel('Time $t$',fontsize=35)
ax.set_ylabel('Position $x$',fontsize=35);
#fig.savefig(figdir + "diff.png")

#%%

