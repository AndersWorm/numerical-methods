import numpy as np
from scipy import linalg as la
A=np.random.rand(5,5)# 20x20 matrix with random numbers between 0 and 1
def powermethod(A,tol):
    v_i=np.random.rand(A.shape[1],1) #since v is a collumn vector which needs to have the same numbers of rows as A has collumns
    u=A@v_i #the initial u
    mu=la.norm(u,np.inf) #initial mu
    v_f=u/mu #the next v
    mu_i=mu+1 #just some starting value to make sure it starts
    while(abs(mu-mu_i)>tol):
        mu_i=mu
        v_i=v_f
        u=A@v_i
        mu=la.norm(u,np.inf)
        v_f=u/mu
    return(v_f,mu)
V,U=powermethod(A,1e-7)
print("V_max=",V,"with eigenvalue U_max=", U)
I= la.norm(A@V-U*V,2) #they must be the same so I must be 0
print("I=", np.round(I))
#smallest eigenvector and value exact same procedure but A is now A^-1
def invpeig(A,tol):
    v_i=np.random.rand(A.shape[1],1) #since v is a collumn vector which needs to have the same numbers of rows as A has collumns
    Ainv = la.inv(A)
    u=Ainv@v_i #the initial u
    mu=la.norm(u,np.inf) #initial mu
    v_f=u/mu #the next v
    mu_i=mu+1 #just some starting value to make sure it starts

    while(abs(mu-mu_i)>tol): #The loop should repeat the process above untill it converges
        mu_i=mu
        v_i=v_f
        u=Ainv@v_i
        mu=la.norm(u,np.inf)
        v_f=u/mu
    return(v_f,1/mu) #since the lowest eigenvalue is 1/mu
V,U=invpeig(A,1e-7)
print("V_min=",V,"with eigenvalue U_min=", U)
I= la.norm(A@V-U*V,2) #they must be the same so I must be 0
print("I=", np.round(I))