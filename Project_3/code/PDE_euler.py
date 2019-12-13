# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:19:11 2019

@author: Eirik Nordgård
"""
import numpy as np
import matplotlib.pyplot as plt

# define inital condition
def initial(x):
    return np.sin(x*np.pi)

def solve(init_func, T, Nt=100, L=1, safety_factor=0.9):

    # Space grid and time parameters

    dt = T/(Nt - 1)
    dx = np.sqrt(2*dt/safety_factor)
    Nx = int(L/dx) + 1
    c = dt/dx**2

    # Create arrays
    x = np.linspace(0, L, Nx)
    u = np.zeros((Nt, Nx))
    u[0, :] = init_func(x)

    # solve explicit differential equations
    # boundry conditions u[0] = u[-1] = 0 are not touched in loop
    for n in range(0, Nt-1):
        for i in range(1, Nx-1):
            u[n+1, i] = c*(u[n, i+1] - 2*u[n, i] + u[n, i-1]) + u[n, i]

    return [u, x]

# define exact solution
def exact(x, t):
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)

def initial(x):
    return np.sin(np.pi*x) 
#%%
#Solving the equation for two dx
"""
u1, x1 = solve(initial,0.5,Nt=100) #Gived dx = 0.1
u1_e = exact(x1,0.5)
in1 = initial(x1)
u2, x2 = solve(initial,0.005,Nt=100) #Gived dx = 0.01
u2_e = exact(x2,0.005)
in2 = initial(x2)
"""
u1, x1 = solve(initial,0.1,Nt=19) #Gived dx = 0.1
u1_e = exact(x1,0.1)
in1 = initial(x1)
u2, x2 = solve(initial,0.1,Nt=1300) #Gived dx = 0.01
u2_e = exact(x2,0.1)
in2 = initial(x2)
u3, x3 = solve(initial,0.6,Nt=125) #Gived dx = 0.1
u3_e = exact(x3,0.6)
in3 = initial(x3)
u4, x4 = solve(initial,0.6,Nt=10000) #Gived dx = 0.01
u4_e = exact(x4,0.6)
in4 = initial(x4)
#%%
T=0.6
L=1
Nt=10000

dt = T/(Nt - 1)
dx = np.sqrt(2*dt/0.95) #To ensure stability
Nx = int(L/dx) + 1

#%%
plt.figure()
plt.subplot(2,1,1)
plt.xlabel("Euler solutions after $\Delta x=0.1$, t=0.1",fontsize=35)
plt.plot(u1[1,:],linewidth=4, label="u, t=0.1 dx=0.1")
plt.plot(u3[1,:],linewidth=4, label="u, t=0.6 dx=0.1")
plt.plot(u1_e,linewidth=4,label="Exact")
plt.plot(in1,linewidth=4,label="Initial")
plt.tick_params(axis='both', which='major', labelsize=30)
plt.legend(fontsize=15)

plt.subplot(2,1,2)
plt.xlabel("Euler solutions after $\Delta x=0.01$, t=0.1",fontsize=35)
plt.plot(u2[90,:],linewidth=4, label="u, t=0.1 dx=0.01")
plt.plot(u4[80,:],linewidth=4, label="u, t=0.6 dx=0.1")
plt.plot(u2_e,linewidth=4,label="Exact")
plt.plot(in2,linewidth=4,label="Initial")
plt.tick_params(axis='both', which='major', labelsize=30)
plt.legend(fontsize=15)
#plt.savefig('figures/euler_dt0.001.png')
plt.show()

#%%
U=[]
xx=[]
for i in np.arange(0.1, 1, 0.1):
    u1, x1 = solve(initial,i,Nt=10)
    U.append(u1)
    xx.append(x1)
plt.figure()
plt.plot(u1[1])
plt.show()    
#%%

# plot exact solution vs computed for some time t
[u1_1, x1] = solve(initial, 0.02, Nt=10)
[u1_2, x2] = solve(initial, 0.3, Nt=10)#[0]

[u2_1, x3] = solve(initial, 0.02, Nt=100)
[u2_2, x4] = solve(initial, 0.3, Nt=100)#[0]

plt.figure()
plt.plot(x2, initial(x2), label="inital")
plt.plot(x1, u1_1[-1, :], label="u, t=0.02, $\Delta x=0.1$")
plt.plot(x3, u2_1[-1, :], label="u, t=0.02, $\Delta x=0.01$")
plt.plot(x2, exact(x2, 0.02), '--', label="$u_e$")
plt.legend()
#%%
from sklearn.metrics import mean_squared_error
mse_1 = mean_squared_error(u1_e,u1[1])
mse_3 = mean_squared_error(u3_e,u3[1])

print(f'MSE NN is {mse_1:.5f}')
print(f'MSE NN is {mse_3:.5f}')

#%%

# plt.figure(2)
# plt.plot(x2, initial(x2), label="inital")
plt.figure()
plt.plot(x2, u1_2[-1, :], label="u, t=0.3, $\Delta x=0.1$")
plt.plot(x4, u2_2[-1, :], label="u, t=0.3, $\Delta x=0.01$")
plt.plot(x2, exact(x2, 0.3), '--', label="$u_e$")
plt.legend()
#plt.savefig("../figures/FD_solved.pdf")
plt.show()
#%%
"""
    # compute MSE of the error for the different cases:
    print("---------For t = 0.02:---------")
    print(f"dx = 0.1  | MSE = {np.mean((u1_1[-1, :]-exact(x1,0.02))**2)}")
    print(f"dx = 0.01 | MSE = {np.mean((u2_1[-1, :]-exact(x2,0.02))**2)}")
    print("---------For t = 0.3-----------")
    print(f"dx = 0.1  | MSE = {np.mean((u1_2[-1, :]-exact(x1,0.3))**2)}")
    print(f"dx = 0.01 | MSE = {np.mean((u2_2[-1, :]-exact(x2,0.3))**2)}")   
"""
#Insert change here 