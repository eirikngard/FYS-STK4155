# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:07:00 2019

@author: Eirik N
"""

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
import time

sns.set()
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(42)
np.random.seed(42)

def f(x, A_tf):
    """
    function denoted as f in the paper by YI et al.
    x is tensor of size (n,1)
    returns tensor of size (n,1)
    """
    I = tf.eye(n,dtype=tf.float64)
    xT = tf.transpose(x)
    term1 = tf.matmul(xT,x)*A_tf
    term2 = (1 - tf.matmul(tf.matmul(xT,A_tf),x))*I
    return tf.matmul((term1 + term2),x)

def compute_eigval(v, A):
    """
    function for computing eigenvalue, given eigenvector v
    v is vector of size n
    returns a float
    """
    v = v.reshape(n,1)
    vT = v.transpose()
    num = np.matmul(np.matmul(vT,A),v)[0,0]
    den = np.matmul(vT,v)[0,0]
    return num/den

def neural_network(A_tf, v0, t_max, dt, n, prec, learning_rate):
    Nt = int(t_max/dt)
    Nx = n
    t = np.linspace(0, (Nt-1)*dt, Nt)
    x = np.linspace(1, Nx, Nx)

    # Create mesh and convert to tensors
    X, T = np.meshgrid(x, t)
    V, T_ = np.meshgrid(v0, t)

    x_ = (X.ravel()).reshape(-1, 1)
    t_ = (T.ravel()).reshape(-1, 1)
    v0_ = (V.ravel()).reshape(-1, 1)

    x_tf = tf.convert_to_tensor(x_,dtype=tf.float64)
    t_tf = tf.convert_to_tensor(t_,dtype=tf.float64)
    v0_tf = tf.convert_to_tensor(v0_,dtype=tf.float64)

    points = tf.concat([x_tf, t_tf], 1)

    #num_iter = 20000
    num_hidden_neurons = [20,20]
    num_hidden_layers = np.size(num_hidden_neurons)

    with tf.name_scope('dnn'):

        # Input layer
        previous_layer = points

        # Hidden layers
        for l in range(num_hidden_layers):
            current_layer = tf.layers.dense(previous_layer, \
                                            num_hidden_neurons[l], \
                                            name='hidden%d'%(l+1), \
                                            activation=tf.nn.sigmoid)
            previous_layer = current_layer

        # Output layer
        dnn_output = tf.layers.dense(previous_layer, 1, name='output')

    #define loss function
    with tf.name_scope('cost'):
        trial = dnn_output*t_tf + v0_tf*k

        # calculate the gradients
        trial_dt = tf.gradients(trial, t_tf)

        #reshape to allow itterating over time stemps
        trial_rs = tf.reshape(trial,(Nt, Nx))
        trial_dt_rs = tf.reshape(trial_dt,(Nt, Nx))

        # calculate cost function, mse
        cost_temp = 0
        for j in range(Nt):
            trial_temp = tf.reshape(trial_rs[j],(n,1))
            trial_dt_temp = tf.reshape(trial_dt_rs[j],(n,1))
            rhs = f(trial_temp, A_tf) - trial_temp
            err = tf.square(-trial_dt_temp+rhs)
            cost_temp += tf.reduce_sum(err)
        cost = tf.reduce_sum(cost_temp/(Nx*Nt), name='cost')
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        traning_op = optimizer.minimize(cost)

    v_dnn = None

    init = tf.global_variables_initializer()


    with tf.Session() as sess:
        # Initialize the whole graph
        init.run()

        # Evaluate the initial cost:
        print('Initial cost: %g'%cost.eval())

        # The training of the network:
        i = 0
        #for i in range(num_iter):
        while cost.eval()>prec:
            sess.run(traning_op)
            i += 1
            # If one desires to see how the cost function behaves for each iteration:
            if i % 1000 == 0:
                print(i,'iter: %g'%cost.eval())
        # Training is done, and we have an approximate solution to the ODE
        print('Final cost: %g'%cost.eval())

        # Store the result
        v_dnn = tf.reshape(trial,(Nt,Nx))
        v_dnn = v_dnn.eval()


        return v_dnn, t, i

#if __name__ == '__main__':


#set to 1 for max eigenvalue, -1 for min eigenvalue
k = +1

#code for generating random, symmetric nxn matrix
n = 6
Q = np.random.rand(n,n)
A = (Q.T+Q)/2
A_temp = k*A
A_tf = tf.convert_to_tensor(A_temp,dtype=tf.float64)

#compute eigenvalues with numpy.linalg, with timing
tic = time.process_time()
w_np, v_np = np.linalg.eig(A)
toc = time.process_time()
time_np = toc-tic
print('Time np: ', time_np)

idx = np.argsort(w_np)
v_np = v_np[:,idx]
w_min_np = np.min(w_np)
w_max_np = np.max(w_np)
v_min_np = v_np[:,0]
v_max_np = v_np[:,-1]

#setting up the NN
prec = 0.0001
t_max = 3
dt = 0.1
learning_rate = 0.001
v0 = np.random.rand(n)

#timing the neural network
tic = time.process_time()
v_dnn, t, i = neural_network(A_tf, v0, t_max, dt, n, prec, learning_rate)
toc = time.process_time()
time_nn = toc-tic

#np.save('../data/eigenvectors/max_vec_dt{:.0E}t_max{:.0f}prec{:.0E}lr{:.0E}'.format(dt,t_max,prec,learning_rate),v_dnn)

print('Time nn: ', time_nn)

#plotting
fig, ax = plt.subplots()
ax.plot(t, v_dnn, color='black')
ax.set_xlabel(r'Time $t$', fontsize=20)
ax.set_ylabel(r'Estimated $v_{max}$ elements', fontsize=20)
#ax.set_ylabel(r'Estimated $v_{min}$ elements', fontsize=20)
ax.text(0.7, 0.95, 'dt = {} \n $\epsilon$ \, = {} \n i \,\, = {}'.format(dt,prec,i) , \
        horizontalalignment='left', verticalalignment='top',\
        transform=ax.transAxes, fontsize = 20)
ax.tick_params(axis='both', labelsize=14)
plt.tight_layout()
figdir = "../figures/"
plt.savefig(figdir + "eigenvalues1.png")
#plt.savefig('../figures/eigenvector_min.pdf')

v_last_dnn = v_dnn[-1]
w_last_dnn = compute_eigval(v_last_dnn, A)
print('v0: \n', v0)
print('v nn: \n',v_last_dnn)
print('unit v nn: \n', v_last_dnn/np.linalg.norm(v_last_dnn))
print('unit v max np: \n',v_max_np)
print('unit v min np: \n',v_min_np)
print('w nn: \n',w_last_dnn)
print('w max numpy: \n',w_max_np)
print('w min numpy: \n',w_min_np)
print('time nn/time np: ', time_nn/time_np)

plt.show()