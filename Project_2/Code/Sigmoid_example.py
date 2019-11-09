# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 11:22:07 2019

@author: Eirik N
"""

"""The sigmoid function (or the logistic curve) is a
function that takes any real number, z, and outputs a number (0,1).
It is useful in neural networks for assigning weights on a relative scale.
The value z is the weighted sum of parameters involved in the learning algorithm."""

import numpy
import matplotlib.pyplot as plt
import math as mt

z = numpy.arange(-5, 5, .1)
sigma_fn = numpy.vectorize(lambda z: 1/(1+numpy.exp(-z)))
sigma = sigma_fn(z)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z, sigma,linewidth=3,color='g')
ax.set_ylim([-0.1, 1.1])
ax.set_xlim([-5,5])
ax.grid(True)
ax.tick_params(labelsize=15)
ax.set_xlabel('z',size=15)
ax.set_ylabel('$\sigma$',size=15)
ax.set_title('Sigmoid function',size=20)

plt.show()