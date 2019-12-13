# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:54:28 2019

@author: Eirik Nordgård
"""

"""
Using Neural Network to solve equation
u_xx = u_t
for a given inital condition u(x,0) = I(x) and
boundries u(0,t) = u(L,t) = 0
using trial funcition:
g_trial(x,t) = (1-t)I(x) + x(1-x)t*N(x,t,P)
N is output from neural network for input x and weights P
"""
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# define initial condition
def initial(x):
    return tf.sin(np.pi*x)

def exact(x, t):
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)

def solve(init_func, T=0.08, Nx=100, Nt=10, L=1, learning_rate=1e-3, num_iter=1e4):
    tf.set_random_seed(4155)
    # resetting neural network
    tf.reset_default_graph()
    # Defining grid interval
    dx = L/(Nx - 1)

    dt = T/(Nt - 1)

    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T, Nt)

    # Create mesh and convert to tensors
    X, T = np.meshgrid(x, t)

    x_ = (X.ravel()).reshape(-1, 1)
    t_ = (T.ravel()).reshape(-1, 1)

    x_tf = tf.convert_to_tensor(x_)
    t_tf = tf.convert_to_tensor(t_)

    points = tf.concat([x_tf, t_tf], 1)

    # SET UP NEURAL NETWORK

    num_hidden_neurons = [30,30]
    num_hidden_layers = np.size(num_hidden_neurons)

    with tf.variable_scope('nn', reuse=tf.AUTO_REUSE):
        # input layer
        previous_layer = points

        # hidden layers
        for l in range(num_hidden_layers):
            current_layer = tf.layers.dense(previous_layer,
                                            num_hidden_neurons[l],
                                            activation=tf.nn.sigmoid)
            previous_layer = current_layer

        # output layer
        nn_output = tf.layers.dense(previous_layer, 1)

    # set up cost function (error^2)

    with tf.name_scope('cost'):
        # define trial funcition
        trial = (1-t_tf)*initial(x_tf) + x_tf*(1-x_tf)*t_tf*nn_output

        # calculate the gradients
        trial_dt = tf.gradients(trial, t_tf)
        trial_d2x = tf.gradients(tf.gradients(trial, x_tf), x_tf)

        # calculate cost function
        err = tf.square(trial_dt[0] - trial_d2x[0])
        cost = tf.reduce_sum(err, name='cost')

    # define learning rate and minimization of cost function
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(cost)

    # definie itialization of all nodes
    init = tf.global_variables_initializer()

    # define a storage value for the solution
    u_nn = None

    # CALCULATE AND SOLVE THE PDE
    with tf.Session() as session:
        # Initialize the computational graph
        init.run()

        #print('Initial cost: %g'%cost.eval())

        for i in range(int(num_iter)):
            session.run(training_op)

        #print('Final cost: %g'%cost.eval())

        u_nn = trial.eval()

    # define exact solution

    # reshape arrays
    U_nn = u_nn.reshape((Nt, Nx))
    return U_nn, x, t


if __name__ == "__main__":
    u, x, t = solve(initial, T=0.02)



    print(f"MSE = {np.mean((u[-1, :]-exact(x, t[-1]))**2)}")

    plt.figure(1)
    plt.plot(x, u[-1, :], label=f"Neural Network, t = {t[-1]}")
    plt.plot(x, exact(x, t[-1]), label=f"u_e, t = {t[-1]}")
    plt.plot(x, u[5, :], label=f"Neural Network, t = {t[5]}")
    plt.plot(x, exact(x, t[5]), label=f"u_e, t = {t[5]}")
    plt.legend()
    #plt.savefig("../figures/NN_solved.pdf")
    plt.show()