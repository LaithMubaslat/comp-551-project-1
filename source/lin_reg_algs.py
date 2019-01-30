import numpy as np


def closed_form(X, y):
    X_T_X = np.dot(X.T,X)
    weights = np.dot(np.linalg.inv(X_T_X), np.dot(X.T, y))
    return weights


def gradient_descent(X, y, W, beta, eta, epsilon):
    X_T_X=np.dot(X.T,X)
    W_old=W
    while (True):
        alpha= eta / (1 + beta)
        W_new=W_old-2*alpha*np.subtract(np.dot(X_T_X,W_old), np.dot(X.T, y))
        #Stopping Criteria
        if np.linalg.norm(np.subtract(W_new,W_old))<epsilon:
            print ("Converged!")
            break
        else:
            W_new=W_old
            beta= beta * 2 #multiply by any cst to get alpha very small approx 10^-7
    return W_new

         
def ridge_regression(X,Y, lamb):
    coefficients = np.dot(np.linalg.inv(np.dot(X.T,X)+np.diag(lamb*np.ones(123))),np.dot(X.T,Y))
    return coefficients



            


    
    
    
    
    
    
    
    
    
    
    
    
