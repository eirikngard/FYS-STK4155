# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 19:04:57 2019

@author: Eirik N
"""

# RESAMPLING METHODS: BOOTSTRAP

# Boostraping is a nonparametric approach to statistical 
#interference that substitutes cpomutation for mre traditiional distributional assumtions
#and asumptotic results. Advantages:
#1:Its quite general
#2:Since it does not require distributional assumtions (such as normally distributed errors)
#, it can provide more accurate interferences when the data are not well behaved
#or when the sample size is small
#3:It is possible to apply it to statistics with sampling distributions that are
#difficult to derive, even asymptotically
#4: It is relatively simple to apply it to omplex data-collection plans 

#EXS: Gaussian distribution, mean=100, variance=15 used to genrerate data.

from numpy import *
from numpy.random import randint, randn
from time import time
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Returns mean of bootstrap samples                                                                                                                                                
def stat(data):
    return mean(data)

# Bootstrap algorithm
def bootstrap(data, statistic, R):
    t = zeros(R); n = len(data); inds = arange(n); t0 = time()
    # non-parametric bootstrap         
    for i in range(R):
        t[i] = statistic(data[randint(0,n,n)])

    # analysis    
    print("Runtime: %g sec" % (time()-t0)); print("Bootstrap Statistics :")
    print("original           bias      std. error")
    print("%8g %8g %14g %15g" % (statistic(data), std(data),mean(t),std(t)))
    return t


mu, sigma = 100, 15
datapoints = 10000
x = mu + sigma*random.randn(datapoints)
# bootstrap returns the data sample                                    
t = bootstrap(x, stat, datapoints)
# the histogram of the bootstrapped  data                                                                                                    
n, binsboot, patches = plt.hist(t, 50, normed=1, facecolor='red', alpha=0.75)

# add a 'best fit' line  
#Densisty was normpdf
y = mlab.density( binsboot, mean(t), std(t))
lt = plt.plot(binsboot, y, 'r--', linewidth=1)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.axis([99.5, 100.6, 0, 3.0])
plt.grid(True)

plt.show()