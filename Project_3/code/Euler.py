# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:35:40 2019

@author: Eirik Nordgård
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# define inital condition
def initial(x):
    return np.sin(x*np.pi)

def solve(initial_func, T, Nt=100, L=1):
    """
    solves equation u_xx = u_t for a given inital condition and
    boundries u(0,t) = u(L,t) = 0
    """
    # establish space grid and time parameters
    
    dt = T/(Nt - 1) # T final time
    dx = np.sqrt(2*dt/0.95) # Step size. "safetyfactor" to ensure stability 
    Nx = int(L/dx) + 1 #L spatial endpoint

    c = dt/dx**2

    # create arrays
    x = np.linspace(0, L, Nx)
    u = np.zeros((Nt, Nx))
    u[0, :] = initial_func(x) #Function of x at t=0

    # solve explicit differential equations
    # boundry conditions u[0] = u[-1] = 0 are not touched in loop
    for n in range(0, Nt-1):
        for i in range(1, Nx-1):
            u[n+1, i] = c*(u[n, i+1] - 2*u[n, i] + u[n, i-1]) + u[n, i]

    return [u, x]

# define exact solution
def exact(x, t):
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)
#%%
sns.set()
sns.set_style("whitegrid")
sns.set_palette("Blues")
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

figdir = "../figures/"

# plot exact solution vs computed for some time t
# Using Nt = 2*T/(0.9 dx^2) + 1 to get desired dx
u1, x1 = solve(initial, 0.02, Nt=int(2*0.02/(0.9*(0.01)**2) + 1)) #dx = 1/100
u2, x2 = solve(initial, 0.2, Nt=int(2*0.2/(0.9*(0.01)**2) + 1)) #dx = 1/100
u3, x3 = solve(initial, 0.02, Nt=int(2*0.02/(0.9*(0.1)**2) + 1)) #dx = 1/10
u4, x4 = solve(initial, 0.2, Nt=int(2*0.2/(0.9*(0.1)**2) + 1)) #dx = 1/10

fig, ax = plt.subplots(1, 1)

ax.plot(x1, u1[-1, :], color="b", ls="dashed", label="dx=0.01")
ax.plot(x3, u3[-1, :], color="b", ls=":", label="dx=0.1")
ax.plot(x1, exact(x1, 0.02), color="b", ls="dotted", lw=4, label="Exact")
ax.plot(x2, u2[-1, :], color="r", linestyle="dashed")
ax.plot(x4, u4[-1, :], color="r", ls=":")
ax.plot(x2, exact(x2, 0.2), color="r", linestyle="dotted", lw=4)

ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("u(x, t)", fontsize=20)
fig.legend(ncol=3, loc="upper center", frameon=False, fontsize=15)
plt.savefig(figdir + "euler.pdf")
plt.show()

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


