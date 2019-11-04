# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 08:53:16 2019

@author: Eirik N
"""

#Under own code for OLS:
'''
def mse(y,y_tilde):
    return np.sum(y**2-y_tilde**2)/len(y) #Definisjon av MSE
    
result=mse(y2,y_tilde) #Dette gir scoren min for prediksjonen, så nære 0 som mulig 
print("Mean squared error (OLS own code): %.12f" % result)
'''
#%%
'''
Variance and confidense intervall for Beta
'''
#B_OLS=np.linalg.inv(X.T.dot(X).X.T.z) 
#var=np.diagonal(coefs)#Definisjon av variansen til Betaene fra oppgavesett

#var er variansen til alle betaene, i dette tilfelle 3 stykker

#FOR RIDGE REGRESSION

#identity=np.eye(ncols)
#B_r=np.linalg.inv(X.T.dot(X)-lambd*identity).dot(X.T).dot(y)
# Likning 3.44 i The Elements of Statistical Learning
#var=np.diagonal(B_r)
#%%
#%%
'''
def k_f_c_v(x, y, z, reg, degree=5, hyperparam=0, k=5):
    """
    k-fold CV calculating evaluation scores: MSE, R2, variance, and bias for
    data trained on k folds.
    where
        x, y = coordinates (will generalise for arbitrary number of parameters)
        z = data/model
        reg = regression function reg(X, data, hyperparam)
        degree = degree of polynomial
        hyperparam = hyperparameter for calibrating model
        k = number of folds for cross validation
    """
    MSE = []
    R2 = []
    VAR = []
    BIAS = []
    
    
    #bias=np.zeros(20)

    #Shufler dataene 
    x_shuffle, y_shuffle, z_shuffle = shuffle(x, y, z)

    #Splitter dataene i k folder
    x_split = np.array_split(x_shuffle, k)
    y_split = np.array_split(y_shuffle, k)
    z_split = np.array_split(z_shuffle, k)

    #looper over alle forldene
    for i in range(k):
        #Plukker ut test folden fra dataene. Lar hver fold bli trukket som test fold.
        x_test = x_split[i]
        y_test = y_split[i]
        z_test = z_split[i]

        # Setter sammen resten av foldene/settene til train data.
        # concatenate setter samme en sekevens av arrays til en array
        # ravel flater ut araayen
        x_train = np.concatenate(x_split[0:i] + x_split[i+1:]).ravel()
        y_train = np.concatenate(y_split[0:i] + y_split[i+1:]).ravel()
        z_train = np.concatenate(z_split[0:i] + z_split[i+1:]).ravel()

        #fit/tilpasse/ en moel til treningssettet
        X_train = CreateDesignMatrix_X(x_train,y_train,degree)
        beta = reg(X_train, z_train, hyperparam=hyperparam)
        #X_train = generate_design_2Dpolynomial(x_train, y_train, degree=degree)

        #evaluer modellen på test settet 
        X_test = CreateDesignMatrix_X(x_test,y_test,degree)
        z_fit = X_test @ beta
        #X_test = generate_design_2Dpolynomial(x_test, y_test, degree=degree)

        MSE.append(mse(z_test, z_fit)) #mse
        R2.append(r2(z_test, z_fit)) #r2
        BIAS.append(bias(z_test, z_fit))
        VAR.append(variance(z_fit))
        
        #bias = bias + np.mean(bias(z_test,z_fit))
        
        
    return [np.mean(MSE), np.mean(R2), np.mean(BIAS), np.mean(VAR)]

result_least_square=k_f_c_v(x2,y2,z2,least_squares,5,0,5)
result_ridge_regression=k_f_c_v(x2,y2,z2,ridge_regression,5,0,5)
#result_least_square=k_f_c_v(x2,y2,z2,least_squares,5,0,5)
    #return [np.mean(MSE)]
    #return bias

#bias = bias + np.mean(bias)
 #       variance = variance + np.mean(variance)
    #return BIAS
    
#%%
k=5

liste=[]
for i in range(20):
    liste.append(np.array(k_f_c_v(x2,y2,z2,ridge_regression,i,0.1,5)))
    #print(list[1])
liste = np.array(liste)
fig = plt.figure()
plt.plot(liste[:,0], label = 'mse')
plt.plot(liste[:, 2], label = 'Bias')
plt.plot(liste[:, 3], label = 'Var')
plt.plot(liste[:, 3] + liste[:, 2], label = 'Var+bias')
plt.legend()
#plt.title("MSE","BIAS","VARIANCE")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#har nå at var er nesten lik bias lik neste konst. 
'''

#%%
#%%
'''
def least_squares(X, data, hyperparam=0):
    """
    Least squares solved using matrix inversion
    Data is z2 (to get correct shape for data)
    """
    beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(data)
    return beta

    #B_ols=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z.ravel())#fit y.ravel og y.flatten er det samme  
    #y_tilde= X @ beta#prediction, could also write y_tilde=X.dot(B_ols)
    #return beta

#%%
def ridge_regression(X, data, hyperparam=1):
    """
    Ridge regression solved using matrix inversion
    """
    p = len(X[0, :])
    beta = np.linalg.pinv(X.T.dot(X) + hyperparam*np.identity(p)).dot(X.T).dot(data)
    return beta

#%%
   
    def lasso_regression(X, data, hyperparam=1):
    
    #Lasso regression solved using scikit learn's in-built method Lasso
    
    
    #reg = linear_model.Lasso(alpha=hyperparam)
    lass = skl.Lasso(alpha=hyperparam)
    lass.fit(X, data)
    beta = reg.coef_
    return beta


_lambda = 0.0001
clf_lasso = skl.Lasso(alpha=_lambda)#
clf_lasso.fit(X,z2)#Calculates Beta  
ylasso = clf_lasso.predict(X) #Tilnærming til frankefunskjonen 
B_lasso = clf_lasso.coef_ #B0,B1 etc Coefs. of Beta 
intercept = clf_lasso.intercept_ #Gir Beta0
y_tilde_lasso_scikit=ylasso.ravel()
'''

"""
From PIAZZA
"""
#%%
'''
Hi, as you can see, our plots look fine up till the 10th polynomial, but after 
this, our errors get enormous. Is this to be expected? Is it due to numerical 
precision or round-off errors?

This depends on inversion method used.
Using np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z) I got very similar results.
To avoid this problem, you could use an SVD inversion method.
I stole Mortens code from the regression slides,

def ols_svd(X:np.ndarray, z:np.ndarray )->np.ndarray:
    u, s, v = scl.svd(X)
    return v.T @ scl.pinv(scl.diagsvd(s, u.shape[0], v.shape[0])) @ u.T @ z
(or something like this)
Using this inversion method I got up to a 40th order polynomial without getting errors like that. (Could probably get higher, but didn't test.)
If it is easily implemented in the code, try this out.
'''
#%%
'''
Hi,
I find that most literature I've read (Hastie, et al., various wikipedia articles, 
and a few data-science blogs) states that we want to minimize the residual sum of 
squares in order to obtain our optimized β-coefficients. However, in the lecture
 notes, our optimization criterion is defined as:
minβ∈Rp1n{(y−Xβ)T(y−Xβ)}.
Which I read as the mean squared error, not the RSS (the difference being the 1n part).
Am I reading this wrong? Does it make a difference?

In practice you end up with the same equations, the 1/n part disappears. Note 
also that many texts use the expectation value and that leads to the MSE.  
This is also used when you derive the OLS equations from a normal distribution 
or when you use gradient descent and need the precise definition of the gradients
 (which then includes 1/n).
'''
#%%
'''
Could you try to run with  larger span of values for the hyperparameters? 
That is 10−5,10−4,…101, just as an example. Also, switching off the noise may 
be better. For the smooth Franke function it is indeed stadnard OLS which gives
 you the best fit without noise. You should be able then to see a trend in worsening MSE.
 
 Excellent! And probably you see for the Franke function that is OLS which 
 gives the best results (with no noise).

Yes! I guess this is not really unexpected. The franke function is after all a 
smooth function, and when I do the regression on a large amount of points in 
[0,1]2 there's a limit to how wildly the function  can behave between points,
 so the need for regularization should be small, at least compared to the case
 with noise added! 
 
'''
#%%
'''
As replied above, Lasso drives indeed coefficients to zero. For a discussion 
see for example the text of Hastie et al, chapter 3.
'''
#%%
'''
This is a very good question, to which I am not sure if I have a clear answer.
 You will indeed perform many runs which could be included in an appendix or 
 online at your Github or GitLab repo.  It is in a way you who make the decisions. 
 I would personally present the best model (lowest MSE) for a given complexity
 (for Ridge and Lasso this means finding the lambda value with lowest MSE) and
 perhaps just have a table which summarizes the results for different hyper 
 parameters. For Lasso you also have a learning rate.  The wealth of data you 
 produce can then be used to support your conclusions by placing them in say an 
 appendix.  To the question below by Kristian, we do not deduct points if you 
 don't present every single result you have gotten. 
 '''
#%%
'''
For cofidence intervalls:
    As far as I have understood from equation 3.14 in Hastie et al, 
    the spread of β also depends on the variance σ² of the error term. 
    For σ=1, we get an equally large spread. For σ²=0.1, we get a much 
    smaller spread. 
'''
#%%
'''
indeed, in part a) we don't ask for resampling. Later we ask for a resampling 
    of data and we recommend writing a cross-validation function. If you wish 
    to study bootstrap as well, this is just an additional bonus.  But the 
    resampling we recommend (since it is widely used) is CV.
'''
#%%
'''
What is the polynomial degree you try to fit with? The MSE you list sounds very
 large indeed.  However, it depends on the degree of the polynomial. For small
 degrees of the polynomial you use to fit the data, you may have a small MSE 
 for OLS. For Ridge it depends on the value of the hyper parameter λ. You 
 would need to tune it and use the λ value which gives the smallest MSE. For 
 standard OLS, you may quickly get values like the one you list, or even larger
 if you increase the complexity. For this case you will see that it is Lasso 
 which gives the best result.
'''
#%%
'''
From what I can tell from your code, you are not adding any random noise to
 z. One of the reasons for overfitting is that as your model get more and more 
 flexible, it will start fitting to the noise. 
'''
 #%%
 '''
It is always good to have selected calculations that an eventual reader can reproduce. As an example, if we run your code for fitting of the Franke function with your Ridge regression, it is always good to have some test calculations in say the github repository to link to.

As an example, your GitHub repository could contain three folders

1) the code(s)

2) your report

3) test calculations (some selected cases)

That way we can reproduce your results and see that everything is ok when we test your code. Making your science reproducible is an important part of a report (in particular when you write your thesis). It serves also as a benchmark/test when you look up your code in say 6 months. Often we forget what a specific code should produce, having these test calculations with output tells you what you should get if you run it with those input parameters.

'''
#%%
'''
You are probably looking at the MSE and R2 between your prediction and the 
noisy data. What you want is to see whether your prediction is able to reproduce
 the true function (the Franke function without noise). You should try to
 calculate your MSE and R2 between your prediction and the true values
 (unnoised franke function).
'''

#%%

The first is prediction accuracy: the least squares estimates often have
low bias but large variance. Prediction accuracy can sometimes be
improved by shrinking or setting some coefficients to zero. By doing
so we sacrifice a little bit of bias to reduce the variance of the predicted
values, and hence may improve the overall prediction accuracy.
Hastie 3.3.