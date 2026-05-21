import numpy as np
from fwdeuler import forward_euler
from rk2 import rk2

def auto_step_ctrl(fcn, x0, t0, tf, h0, Atol, Rtol):
    """ Parameters : For y' = fcn(t,y)
    fcn - fcn(t,y)
    x0 - initial point
    t0 -initial time
    tf - final time, time span is [t0,tf]
    h0 - starting step size
    Atol - array of desired absolute tolerances
    Rtol - array of desired relative tolerance
    
    Returns : 
    """
    y, T1 = forward_euler(fcn,x0,t0,tf,h0)
    y_hat, T2 = rk2(fcn,x0,t0,tf,h0)

    #print(T1)

    #Compute componentwise error bound
    sc = np.zeros(len(T1))
    
    for i in range(0,len(T1)):
        sc[i] = Atol[i] + (max(abs(y(T1[i])), abs(y_hat(T1[i]))))*Rtol[i]
        #sc[i] = i

    # Compute measure of error
    n = len(T1)+1
    vals = np.zeros(len(T1))

    for i in range(0,len(T1)):
        vals[i] = ((y(T1[i])-y_hat(T1[i]))/sc[i])**2

    err = np.sqrt((1/n)*np.sum(vals))

    return err