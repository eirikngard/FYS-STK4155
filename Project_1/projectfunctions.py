import numpy as np
from sklearn.utils import shuffle
from sklearn import linear_model


def generate_design_polynomial(x, p=1):
    """
    Creates a design matrix for a 1d polynomial of degree p
        1 + x + x**2 + ...
    """
    X = np.zeros((len(x), p+1))
    for degree in range(0, p+1):
        X[:, degree] = (x.T)**degree
    return X

def generate_design_2Dpolynomial(x, y, degree=5):
    """
    Creates a design matrix for a 2d polynomial with cross-elements
        1 + x + y + x**2 + xy + y**2 + ...
    """
    X = np.zeros(( len(x), int(0.5*(degree + 2)*(degree + 1)) ))
    p = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            X[:, p] = x**i*y**j
            p += 1
    return X

def least_squares(X, data):
    """
    Least squares solved using matrix inversion
    """
    beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(data)
    return beta

def ridge_regression(X, data, hyperparam=0):
    """
    Ridge regression solved using matrix inversion
    """
    p = len(X[0, :])
    beta = np.linalg.pinv(X.T.dot(X) + hyperparam*np.identity(p)).dot(X.T).dot(data)
    return beta

def lasso_regression(X, data, hyperparam=1):
    """
    Lasso regression solved using scikit learn's in-built method Lasso
    """
    reg = linear_model.Lasso(alpha=hyperparam)
    reg.fit(X, data)
    beta = reg.coef_
    return beta

def mse(data, model):
    """
    Calculates the mean square error between data and model.
    """
    n = len(data)
    error = np.sum((data - model)**2)/n
    return error

def r2(data, model):
    """
    Calculates the R2-value of the model.
    """
    n = len(data)
    error = 1 - np.sum((data - model)**2)/np.sum((data - np.mean(data))**2)
    return error

def expectation(models):
    """compute a mean vector from n vectors """
    mean_model =  np.mean(models, axis=1, keepdims=True)
    return mean_model

def bias(data, model):
    """caluclate bias from k expectation values and data of length n"""
    n = len(data)
    error = mse(data, np.mean(model))
    return error

def variance(model):
    """
    Calculating the variance of the model: Var[model]
    """
    n = len(model)
    error = bias(model, model)
    return error

def k_fold_cross_validation(x, y, z, reg, degree=5, hyperparam=0, k=5):
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

    #shuffle the data
    x_shuffle, y_shuffle, z_shuffle = shuffle(x, y, z)

    #split the data into k folds
    x_split = np.array_split(x_shuffle, k)
    y_split = np.array_split(y_shuffle, k)
    z_split = np.array_split(z_shuffle, k)

    #loop through the folds
    for i in range(k):
        #pick out the test fold from data
        x_test = x_split[i]
        y_test = y_split[i]
        z_test = z_split[i]

        # pick out the remaining data as training data
        # concatenate joins a sequence of arrays into a array
        # ravel flattens the resulting array
        x_train = np.concatenate(x_split[0:i] + x_split[i+1:]).ravel()
        y_train = np.concatenate(y_split[0:i] + y_split[i+1:]).ravel()
        z_train = np.concatenate(z_split[0:i] + z_split[i+1:]).ravel()

        #fit a model to the training set
        X_train = generate_design_2Dpolynomial(x_train, y_train, degree=degree)
        beta = reg(X_train, z_train, hyperparam=hyperparam)

        #evaluate the model on the test set
        X_test = generate_design_2Dpolynomial(x_test, y_test, degree=degree)
        z_fit = X_test @ beta


        MSE.append(mse(z_test, z_fit)) #mse
        R2.append(r2(z_test, z_fit)) #r2
        BIAS.append(bias(z_test, z_fit))
        VAR.append(variance(z_fit))

    return [np.mean(MSE), np.mean(R2), np.mean(BIAS), np.mean(VAR)]

def frankefunction(x, y, noise=1):
    term1 = 0.75*np.exp(-(0.25*(9*x - 2)**2) - 0.25*((9*y - 2)**2))
    term2 = 0.75*np.exp(-((9*x + 1)**2)/49.0 - 0.1*(9*y + 1))
    term3 = 0.5*np.exp(-(9*x - 7)**2/4.0 - 0.25*((9*y - 3)**2))
    term4 = -0.2*np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    noise_ = np.random.normal(0, noise, len(x))
    return term1 + term2 + term3 + term4 + noise_

def produce_table(data, header):
    """
    Spagetthi code producing a vertical laTEX table.
    data has shape
        [[x0, x1, x2, ..., xN],
         [y0, y1, y2, ..., yN],
         ...          ]
    where
    header = list/array
    """
    tableString = ""
    n = len(data[:, 0])
    tableString += "\\begin{table}[htbp]\n"
    tableString += "\\begin{{tabular}}{{{0:s}}}\n".format("l"*n)
    # creating header
    for element in header:
        tableString += f"\\textbf{{{element}}} & "
    tableString = tableString[:-2] + "\\\\\n"
    # creating table elements
    for j in range(len(data[0, :])):
        for i in range(len(data[:, 0])):
            tableString += f"{data[i, j]:.2f} & "
        tableString = tableString[:-2] + "\\\\\n"
    tableString = tableString[:-4] + "\n"
    tableString += "\\end{tabular}\n"
    tableString += "\\end{table}\n"
    return tableString
