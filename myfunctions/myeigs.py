import numpy as np
from scipy import linalg as la
def peig(A,tol): #biggest eigenvector and eigenvalue
    v_i=np.random.rand(A.shape[1],1) #since v is a collumn vector which needs to have the same numbers of rows as A has collumns
    u=A@v_i #the initial u
    mu=la.norm(u,np.inf) #initial mu
    v_f=u/mu #the next v
    while(la.norm((v_f-v_i),2)>tol): #The loop should repeat the process above untill it converges
        v_i=v_f
        u=A@v_i
        mu=la.norm(u,np.inf)
        v_f=u/mu
    return(v_f,mu)
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
def qreigs(A,tol):
    Q_i,R_i=la.qr(A)
    A_f=R_i@Q_i
    while(la.norm((np.diag(A)-np.diag(A_f)),2)>tol): #because it is only the eigenvalues who needs to converge
        A=A_f
        Q_i,R_i=la.qr(A)
        A_f=R_i@Q_i
    return(np.diag(A_f))