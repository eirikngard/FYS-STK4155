# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 09:45:18 2019

@author: Eirik Nordgård
"""
## Brukt koden https://github.com/krisbhei/DEnet/blob/master/DNN_Diffeq/example_pde_diffusion.ipynb
## som utgangspunkt

import numpy as np 
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import tensorflow.compat.v1 as tf


def neural_network(num_iter=100):
    tf.disable_v2_behavior()

    # Reset the graph 
    tf.reset_default_graph()#running the ocde didnt work, this fixed it 

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
    
    #num_iter = 100000 #was 100000 initially, adjust this later
    
    num_hidden_neurons = [10]#was 30
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
    
U_nn,U_e,t,x,T,X,num_hidden_neurons,u_e,u_nn = neural_network(100000)
diff_mat = np.abs(U_e - U_nn)

# Surface plot of the solutions


T,X = np.meshgrid(t, x)

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Solution from the deep neural network w/ %d layer'%len(num_hidden_neurons),fontsize=35)
s = ax.plot_surface(T,X,U_nn,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_xlabel('Time $t$',fontsize=35)
ax.set_ylabel('Position $x$',fontsize=35);
#fig.savefig('figures\dnn.png')

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Analytical solution',fontsize=35)
s = ax.plot_surface(T,X,U_e,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_xlabel('Time $t$',fontsize=35)
ax.set_ylabel('Position $x$',fontsize=35);
#fig.savefig('figures\exact.png')

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Difference',fontsize=35)
s = ax.plot_surface(T,X,diff_mat,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_xlabel('Time $t$',fontsize=35)
ax.set_ylabel('Position $x$',fontsize=35);
#fig.savefig('figures\diff_dnn_exact.png')

#%%
"""
#2D plot of the solutions
plt.figure(figsize=(10,10))
plt.title("Solutions, analytical by NN and exact ",fontsize=35)
#for i in range(len(U_nn)-1):
plt.plot(x, U_nn[:,2],linewidth=6)
plt.plot(x, U_nn[:,4],linewidth=6)
plt.plot(x, U_nn[:,6],linewidth=6)
plt.plot(x,U_e[:,0],linewidth=6)
plt.legend(["dnn, t = %g"%t[2],"dnn, t = %g"%t[4],"dnn, t = %g"%t[6],\
            'analytical'],fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=30)
#plt.savefig('figures/nn2d.png')

plt.show()
"""
#%%

#Some simple metrics 
from sklearn.metrics import mean_squared_error, r2_score
mse_nn = mean_squared_error(u_e,u_nn)
r2_nn = r2_score(u_e,u_nn)
print(f'MSE NN is {mse_nn:.5f}')
print(f'R2 NN is {r2_nn:.5f}')

mse_Unn = mean_squared_error(U_e,U_nn)
print(f'MSE NN is {mse_Unn:.5f}') #Same as before reshaping, good

total_mse=[]
for i in range(len(u_e)):
    total_mse.append(mean_squared_error(u_e[i],u_nn[i]))
plt.figure()
plt.plot(total_mse,linewidth=6)
plt.title("MSE until t=0",fontsize=35)
plt.ylabel('MSE',fontsize=35)
plt.xlabel('Time',fontsize=35);
#plt.savefig('figures/nn_mse.png')
plt.show()
#%%
sns.set()
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

figdir = "../figures/"


u, x, t = solve(initial, T=0.3, Nt=30)


print("==== For t = 0.3 ====")
print(f"MSE = {np.mean((u[-1, :]-exact(x, t[-1]))**2)}")
print("==== For t = 0.02 ====")
print(f"MSE = {np.mean((u[2, :]-exact(x, t[2]))**2)}")

fig, ax = plt.subplots(1, 1)

ax.plot(x, u[-1, :], color="k", ls="dashed", label="Computed")
ax.plot(x, exact(x, t[-1]), color="k", ls="dotted", lw=4, label="Exact")
ax.plot(x, u[2, :], color="k", ls="dashed")
ax.plot(x, exact(x, t[2]), color="k", ls="dotted", lw=4)

ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("u(x, t)", fontsize=20)

fig.legend(ncol=2, frameon=False, loc="upper center", fontsize=20)
plt.savefig(figdir + "NN_solved.pdf")
plt.show()
