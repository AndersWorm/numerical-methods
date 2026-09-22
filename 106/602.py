import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.special import p_roots

def quads_1(a,b,n,f):
    x,w= p_roots(n)  # x and w are the roots and weights of the Legendre polynomial of degree n
    x_tr=((b-a)/2)*x + (b+a)/2
    return(0.5*(b-a)* np.sum(w*f(x_tr))) #use the formula from the slides

def f(x):
    return(np.exp(x**2))
print(quads_1(-1,1,3,f))

def g(x):
    return(np.exp(-x*2)*np.cos(x))

print(quads_1(1,3,8,g))

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
def h(x):
    return(x**(3)+1)
print(quadz(h,0,2,1e-7))

