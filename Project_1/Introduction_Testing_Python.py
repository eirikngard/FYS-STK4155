"""
Created on Tue Aug 27 10:17:07 2019

@author: Eirik Nordgård
"""
"""
Testing functions from lecture notes
"""
#%%
# Numpy, arrays and matrixes

import numpy as np
n=10
x=np.random.normal(size=n)
x1=np.array([1, 2, 3])
x2=np.log(np.array([1.0, 2.0, 3.0]))
print(x2.itemsize)
A=np.log(np.array([[4.0, 7.0, 8.0], [3.0, 10.0, 11.0], [4.0, 5.0, 7.0] ]))
# print the first column, row-major order and elements start with 0
print(A[:,1]) 
print(A)
B=np.zeros((n,n))
C=np.ones((n,n))
print(C)
D=np.random.rand(n,n)
print(D)
clear all 
#%%
n=100
x=np.random.normal(size=n)
print("x_mean= ",np.mean(x))
y=4+3*x+np.random.normal(size=n)
print("y_mean= ",np.mean(y))
z=x**3+np.random.normal(size=n)
print("z_mean= ",np.mean(z))
W=np.vstack((x,y,z))
Sigma=np.cov(W)
print(Sigma)
Eigva, Eigve=np.linalg.eig(Sigma)
print(Eigva)
print(Eigve)
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
eye = np.eye(4)
print(eye)
sparse_mtx = sparse.csr_matrix(eye)
print(sparse_mtx)
x = np.linspace(-10,10,100)
y = np.sin(x)
plt.plot(x,y,marker='x')
plt.show()
#%%
"""
## Meet the Pandas (panel data)
"""
#%%
import pandas as pd
from IPython.display import display 
data={'Fornavn': ["Eirik", "Eline","Henning","Ola"], 
      'Etternavn': ["Nordgård","Løkken","Straume","Nordmann"],
      'Fødested':["Veståsen","Øståsen","Bergen","Bodø"],
      'Bursdag':[1997,1993,1992,2000]}
# Doing this: data_pandas=pd.DataFrame(data) with different indexing below:
data_pandas=pd.DataFrame(data,index=[1,2,3,4])
display(data_pandas)
display(data_pandas.loc[1])
display(data_pandas.loc[1:2])
#%%
# Setting up a 10x5 matrix
rows=10
cols=5
a=np.random.randn(rows,cols)
print(a)
df=pd.DataFrame(a)
display(df)
print(df.mean())
print(df.std())
display(df**2)

df.columns=['Første','Andre','Tredje','Fjerde','Femte']
df.index=np.arange(10)
display(df)
print(df['Andre'].mean()) #Skriver ut mean av kol kalt 'Andre'
print(df.info())
print(df.describe()) #Gir mange parametere for matrisen df (means,std,max,min osv)
#%%

from pylab import plt, mpl
plt.style.use('seaborn')
mpl.rcParams['font.family']='serif'

df.cumsum().plot(lw=5.0, figsize=(10,6))#lw er tykkelse på streker, 10x6=lxb
plt.show()

df.plot.bar(figsize=(10,6),rot=15) #Rot roterer tall på x-akse gitt ant grader
#bar plotter verdiene i matrisen df som barer for hver rekke
plt.show()

#%%
#Produce a 4x4 matrix

b=np.arange(16).reshape((4,4))
print(b)
df1=pd.DataFrame(b)
print(df1)
#%%

'''
Reading Data and Fitting
'''
#%%
# Simple Linear Regression using SCIKIT-LEARN

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.random.rand(100,1)
y = 2*x+0.01*np.random.randn(100,1)
linreg = LinearRegression()
linreg.fit(x,y)
xnew = np.array([[0],[1]])
ypredict = linreg.predict(xnew)

plt.plot(xnew, ypredict, "m-")#m er farge 
plt.plot(x, y ,'bo')# b er blue, o er prikker
plt.axis([0,1.0,0, 5.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Simple Linear Regression')
plt.show()
#%%

x = np.random.rand(100,1)
y = 5*x+0.01*np.random.randn(100,1)
linreg = LinearRegression()
linreg.fit(x,y)
ypredict = linreg.predict(x)

plt.plot(x, np.abs(ypredict-y)/abs(y), "go")
#plt.plot(x,y,'b')
plt.axis([0,1.0,0.0, 0.5])
plt.xlabel(r'$x$')
plt.ylabel(r'$\epsilon_{\mathrm{relative}}$')
plt.title(r'Relative error')
plt.show()
#%%
'''
Functionality of scikit_learn
'''
#%%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_squared_log_error,mean_absolute_error

# R^2 score provides a measure of how well future samples are 
#likely to be predicted by the model. (1 to -1, 1 i best)
x=np.random.rand(100,1)
y=2.0+5*x+0.5*np.random.randn(100,1)
linreg=LinearRegression()
linreg.fit(x,y)
ypredict=linreg.predict(x)
print('The intercept alpha: \n',linreg.intercept_)
print('Coefficient beta: \n',linreg.coef_)
# The mean squared error 
print("Mean squared error: %.2f" % mean_squared_error(y,ypredict))
#Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y,ypredict))
#Mean squared log error 
print('Mean squared log error: %.2f' % mean_squared_log_error(y,ypredict))
#Mean absolute error 
print('Mean absolute error: %.2f' % mean_absolute_error(y,ypredict))
plt.plot(x,ypredict,"r-")
plt.plot(x,y,"ro")
plt.axis([0.0,1.0,1.5,7.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Linear Regression fit')
plt.show()
#%%

#Cubic polynomial 

import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

x=np.linspace(0.02,0.98,200)
noise = np.asarray(random.sample((range(200)),200))
y=x**3*noise
yn=x**3*100
poly3 = PolynomialFeatures(degree=3)
X = poly3.fit_transform(x[:,np.newaxis])
clf3 = LinearRegression()
clf3.fit(X,y)

Xplot=poly3.fit_transform(x[:,np.newaxis])
poly3_plot=plt.plot(x, clf3.predict(Xplot), label='Cubic Fit')
plt.plot(x,yn, color='red', label="True Cubic")
plt.scatter(x, y, label='Data', color='orange', s=15)
plt.legend()
plt.show()
#%%
def error(a):
    for i in y:
        err=(y-yn)/yn
    return abs(np.sum(err))/len(err)

print(error(y))
#%%