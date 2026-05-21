import numpy as np
from scipy.interpolate import CubicSpline

def rk2(fcn, x0, t0, tf, h):
    
    """Parameters: For y' = fcn(t,y)
    fcn - fcn(t,y)
    x0 - initial point
    t0 - initial time
    tf - final time,  [t0,tf] is time span
    h - fixed step size

    Returns : interpolated solution y(t), array of time values T
    """
    
    step = (tf-t0)/h    
    X = np.zeros(int(step)+1)
    T = np.zeros(int(step)+1)
    X[0]=x0
    T[0]=t0

    n=0
    while T[n] < tf-h/2:
        t_temp = T[n] + h
        y = X[n] + h*fcn(T[n], X[n])
        X[n+1] = X[n] +0.5*h*(fcn(T[n], X[n]) + fcn(t_temp, y))
        n=n+1
        T[n] = t_temp

    y = CubicSpline(T,X)
    return y, T