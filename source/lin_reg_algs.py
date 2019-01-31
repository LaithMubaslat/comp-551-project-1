import numpy as np


def closed_form(X, y):
    X_T_X = np.dot(X.T,X)
    w = np.dot(np.linalg.inv(X_T_X), np.dot(X.T, y))
    return w


def gradient_descent(X, y, alpha=1e-6, epsilon=1e-6, iterations=1000):
    w= np.random.rand(len(X.columns))
    X_T_X=np.dot(X.T,X)
    w_old=w
    i=0
    while (True):
        i=i+1
        w_new=w_old-2*alpha*np.subtract(np.dot(X_T_X,w_old), np.dot(X.T, y))
        #Stopping Criteria
        if np.linalg.norm(np.subtract(w_new,w_old))<epsilon or i>iterations:
            break
        else:
            w_old=w_new
    return w_new

         
def ridge_regression(X, y, lamb):
    coefficients = np.dot(np.linalg.inv(np.dot(X.T,X) + np.diag(lamb*np.ones(len(y)))), np.dot(X.T, y))
    return coefficients



            


    
    
    
    
    
    
    
    
    
    
    
    
