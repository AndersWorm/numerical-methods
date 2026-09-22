import numpy as np
import matplotlib.pyplot as plt
def polyx(a,x): #creates a polynomial function which sums up all the coefficients.
            f=0 
            for i in range(len(a)): # I use the index to pair the coeficients with x, by using len(a) it can be used for n type polynomials.
                f+=a[i]* x**(i)
            return f
def vandint(x,xp,yp):    
        n=len(xp)
        V=np.zeros((n,n)) #nXn dimensioned matrix with zeroes I can update
        for j in range(n): # I sum over the rows with j constant as V says, find it much easier than Hstack.
            for i in range(n):
                V[i,j]=xp[i,0]**(j) #xp[i,0] means first index in each row i
        VI=np.linalg.inv(V)
        C=np.matmul(VI,yp)
        return(polyx(C,x))
def lagx(x,xp):
    l_j_data={}
    for xj in xp:
        l_j=1
        for xi in xp:
            if xi==xj:
                continue
            l_j*= (x-xi)/(xj-xi)
        l_j_data[xj]=l_j
    return(l_j_data)
def lagint(x,xp,yp):
    L=0
    lag=lagx(x,xp) #get all my lagrange functions
    for j, (xj, lj) in enumerate(lag.items()): # enumerate gives me the index j and the xj,lj values
        L += lj * yp[j] # use the index j to get the corresponding y value from yp
    return(L)