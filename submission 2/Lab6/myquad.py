import numpy as np
import scipy as sp
from scipy.special import p_roots

def qtrapn(f,a,b,N,):
    h = (b-a)/N # interval steps
    I=0 #make sure it gives the same every time
    for i in range(N+1): #+1 to include the end point
        x=a+i*h
        if i==0 or i==(N): 
            I+=f(x) #make sure the end points are not counted double
        else:
            I+=2*f(x) # the rest are
    I = I*h/2
    return I
def qsimpn(f,a,b,N):
    h = (b-a)/N
    I=0
    for i in range(N+1):
        x=a+i*h
        if i==0 or i==(N): #the end points
            I+=f(x)
        elif i%2==0: #all even points are multiplied by two
            I+=2*f(x)
        else:
            I+=4*f(x)   #all uneven points
    I = I*h/3
    return I

def errf(N_i,f,a,b):
    return(abs(qtrapn(f,a,b,N_i)-qtrapn(f,a,b,2*N_i)))
#calculates the difference between the two last points

def qtrapz(f,a,b,tol):
    N_i=1 #starting value for N
    iterations=0
    err=errf(N_i,f,a,b) 
    while err>tol:
        N_i*=2 #double N each time
        err=errf(N_i,f,a,b)
        iterations+=1
        if iterations>15: #just to avoid infinite loops
            print("Warning: Maximum number of iterations reached. The desired tolerance may not have been achieved.")
            break
    return("Number of intervals: {}, Result: {}, Tolerance: {}".format(N_i, qtrapn(f,a,b,N_i), tol))


def quads_1(a,b,n,f):
    x,w= p_roots(n)  # x and w are the roots and weights of the Legendre polynomial of degree n
    x_tr=((b-a)/2)*x + (b+a)/2
    return(0.5*(b-a)* np.sum(w*f(x_tr))) #use the formula from the slides


def quadz(f, a, b, tol, n_max=200):
    n = 1
    I_i = quads_1(a, b, n, f)
    I_f = quads_1(a, b, n+1, f)
    while abs(I_i - I_f) > tol:
        n += 1
        if n >= n_max:
            raise RuntimeError(f"didnot reach={tol} with n={n_max}")
        I_i = I_f
        I_f = quads_1(a, b, n+1, f)
    return I_f, n+1