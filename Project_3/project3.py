# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:52:23 2019

@author: Eirik Nordgård
"""

"""
Based on code from 
https://github.com/krisbhei/DEnet/blob/master/DNN_Diffeq/
example_ode_poisson.ipynb

Using Neural Network to solve equation u_xx = u_t for a given inital 
condition u(x,0) = I(x) and boundries u(0,t) = u(L,t) = 0 using trial 
funcition: g_trial(x,t) = (1-t)I(x) + x(1-x)t*N(x,t,P).
N is output from neural network for input x and weights P
"""
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(4155)

# Decide grid size for space and time
L = 1
dx = 0.1
Nx = int(L/dx) + 1

final_t = 0.08
dt = 0.005
Nt = int(final_t/dt) + 1

x = np.linspace(0, L, Nx)
t = np.linspace(0, final_t, Nt)

# Create mesh and convert to tensors
X, T = np.meshgrid(x, t)

x_ = (X.ravel()).reshape(-1, 1)
t_ = (T.ravel()).reshape(-1, 1)

x_tf = tf.convert_to_tensor(x_)
t_tf = tf.convert_to_tensor(t_)

points = tf.concat([x_tf, t_tf], 1)

# SET UP NEURAL NETWORK
num_iter = 10000

num_hidden_neurons = [30,30]
num_hidden_layers = np.size(num_hidden_neurons)

with tf.variable_scope('nn'):
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
# define initial condition
def initial(x):
    return tf.sin(np.pi*x)

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
learning_rate = 0.001
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    traning_op = optimizer.minimize(cost)

# definie itialization of all nodes
init = tf.global_variables_initializer()

# define a storage value for the solution
u_nn = None

# CALCULATE AND SOLVE THE PDE
with tf.Session() as session:
    # Initialize the computational graph
    init.run()

    print('Initial cost: %g'%cost.eval())

    for i in range(num_iter):
        session.run(traning_op)

    print('Final cost: %g'%cost.eval())

    u_nn = trial.eval()

# define exact solution
u_e = np.exp(-np.pi**2*t_)*np.sin(np.pi*x_)

# reshape arrays
U_nn = u_nn.reshape((Nt, Nx))
U_e = u_e.reshape((Nt, Nx))

print(f"For t = {final_t} and dx = {dx}")
print(f"MSE = {np.mean((U_nn[-1, :]-U_e[-1,:])**2)}")
#%%
plt.figure(1)
plt.plot(x, U_nn[-1, :], label=f"Neural Network, t = {final_t}")
plt.plot(x, U_e[-1, :], label=f"u_e, t = {final_t}")
plt.plot(x, U_nn[5, :], label=f"Neural Network, t = {t[5]}")
plt.plot(x, U_e[5, :], label=f"u_e, t = {t[5]}")
plt.legend()
#plt.savefig("../figures/NN_solved.pdf")
plt.show()
#%%
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

print(mse(U_nn,U_e))
#%%

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

sns.set()
sns.set_style("white")
sns.set_palette("Set2")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(4155)
np.random.seed(4155)


#code for generating random, symmetric nxn matrix
n = 6
Q = np.random.rand(n,n)
A = (Q.T+Q)/2
#change sign of A to mind other eigenvalue
A_tf = tf.convert_to_tensor(A,dtype=tf.float64)

#compute eigenvalues with numpy.linalg
w_np, v_np = np.linalg.eig(A)
idx = np.argsort(w_np)
v_np = v_np[:,idx]
w_min_np = np.min(w_np)
w_max_np = np.max(w_np)
v_min_np = v_np[:,0]
v_max_np = v_np[:,-1]

def f(x):
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

def compute_eigval(v):
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



#setting up the NN
Nt = 20
Nx = n
t = np.linspace(0, 1, Nt) #maa gaa fra 0 til 1 for aa faa konvergens
x = np.linspace(1, Nx, Nx)
v0 = np.random.rand(n)

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

num_iter = 10000
num_hidden_neurons = [30,30]
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
#DETTE MAA ORDNES
#trial solution maa defineres annerledes tror AL
with tf.name_scope('cost'):
    trial = dnn_output*t_tf + v0_tf
    #v0_tf*dnn_output**(-t_tf)#(1-t_tf)*v0_tf + t_tf*dnn_output

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
        rhs = f(trial_temp) - trial_temp
        err = tf.square(-trial_dt_temp+rhs)
        cost_temp += tf.reduce_sum(err)
    cost = tf.reduce_sum(cost_temp/(Nx*Nt), name='cost')
learning_rate = 0.001
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
    for i in range(num_iter):
        sess.run(traning_op)

        # If one desires to see how the cost function behaves for each iteration:
        if i % 1000 == 0:
            print(i,' iterations:', cost.eval())

    # Training is done, and we have an approximate solution to the ODE
    print('Final cost: %g'%cost.eval())

    # Store the result
    #v_dnn_tf = trial.eval()
    v_dnn = tf.reshape(trial,(Nt,Nx))
    v_dnn = v_dnn.eval()

fig, ax = plt.subplots()
ax.plot(v_dnn, color='black')
ax.set_xlabel('Number of timesteps')
ax.set_ylabel('Value of the elements of the estimated eigenvector')

v_max_dnn = v_dnn[-1]
w_max_dnn = compute_eigval(v_max_dnn)
print('v0: \n', v0)
print('v nn: \n',v_max_dnn)
print('unit v nn: \n', v_max_dnn/np.linalg.norm(v_max_dnn))
print('unit v max np: \n',v_max_np)
print('unit v min np: \n',v_min_np)
print('w nn: \n',w_max_dnn)
print('w max numpy: \n',w_max_np)
print('w min numpy: \n',w_min_np)

plt.show()