# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:37:27 2019

@author: Eirik Nordgård
"""

"""
Using Neural Network to solve equation
u_xx = u_t
"""
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# define initial condition
def initial(x):
    return tf.sin(np.pi*x)

def exact(x, t):
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)

def solve(init_func, T=0.08, Nx=100, Nt=10, L=1, learning_rate=1e-3, num_iter=1e3):
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

    # setup of neural network

    num_hidden_neurons = [20,20]
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

    # set up cost function 

    with tf.name_scope('cost'):
        # define trial funcition
        trial = (1-t_tf)*initial(x_tf) + x_tf*(1-x_tf)*t_tf*nn_output

        # calculate gradients
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

    # storage value for the solution
    u_nn = None

    # solve pde
    with tf.Session() as session:
        # Initialize the computational graph
        init.run()
        #print('Initial cost: %g'%cost.eval())
        for i in range(int(num_iter)):
            session.run(training_op)
        #print('Final cost: %g'%cost.eval())
        u_nn = trial.eval()

    # reshape arrays
    U_nn = u_nn.reshape((Nt, Nx))
    return U_nn, x
#%%
"""
sns.set()
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

figdir = "../figures/"

u, x = solve(initial, T=0.2, Nt=30)

print("==== For t = 0.2 ====")
print(f"MSE = {np.mean((u[-1, :]-exact(x, t[-1]))**2)}")
print("==== For t = 0.02 ====")
print(f"MSE = {np.mean((u[2, :]-exact(x, t[2]))**2)}")

fig, ax = plt.subplots(1, 1)

ax.plot(x, u[-1, :], color="b", ls="dashed", label="Computed")
ax.plot(x, exact(x, t[-1]), color="b", ls="dotted", lw=4, label="Exact")
ax.plot(x, u[2, :], color="r", ls="dashed")
ax.plot(x, exact(x, t[2]), color="r", ls="dotted", lw=4)

ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("u(x, t)", fontsize=20)
fig.legend(ncol=2, frameon=False, loc="upper center", fontsize=20)
#plt.savefig(figdir + "nn.png")
plt.show()
"""
#%%
#plot solutions 

figdir = "../figures/"
u1, x1 = solve(initial, 0.01, Nt=10) #dx = 1/100
u2, x2 = solve(initial, 0.3, Nt=10) #dx = 1/100
u3, x3 = solve(initial, 0.01, Nt=10) #dx = 1/10
u4, x4 = solve(initial, 0.3, Nt=10) #dx = 1/10

fig, ax = plt.subplots(1, 1)

ax.plot(x1, u1[-1, :], color="b", ls="dashed", label="dx=0.01")
ax.plot(x3, u3[-1, :], color="b", ls=":", label="dx=0.1")
ax.plot(x1, exact(x1, 0.01), color="b", ls="dotted", lw=4, label="Exact")
ax.plot(x2, u2[-1, :], color="r", linestyle="dashed")
ax.plot(x4, u4[-1, :], color="r", ls=":")
ax.plot(x2, exact(x2, 0.3), color="r", linestyle="dotted", lw=4)

ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("u(x, t)", fontsize=20)
fig.legend(ncol=3, loc="upper center", frameon=False, fontsize=15)
#plt.savefig(figdir + "NN.png")
plt.show()

#OBS!! This plot gives negative solution in some parts of the domain. STRANGE.
#dont know what causes this.
#%%
# compute MSE of the error for the different cases:
print("")
print("---------For t = 0.02:---------")
print(f"dx = 0.01 | MSE = {np.mean((u1[-1, :]-exact(x1,0.02))**2)}")
print("---------For t = 0.2-----------")
print(f"dx = 0.01 | MSE = {np.mean((u2[-1, :]-exact(x2,0.2))**2)}")
print("---------For t = 0.02:---------")
print(f"dx = 0.1 | MSE = {np.mean((u3[-1, :]-exact(x3,0.02))**2)}")
print("---------For t = 0.2-----------")
print(f"dx = 0.1 | MSE = {np.mean((u4[-1, :]-exact(x4,0.2))**2)}")

#%%
#plot 3D solutions 

Nx = 100; Nt = 10
x = np.linspace(0, 1, Nx) 
t = np.linspace(0,1,Nt)

U_nn,x = solve(initial,T=0.1)

figdir = "../figures/"

# Surface plot of the solutions

X,T = np.meshgrid(t, x)

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

fig.show()
