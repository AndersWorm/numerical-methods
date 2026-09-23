import numpy as np
from scipy import linalg as la
def qreigs(A,tol):
    Q_i,R_i=la.qr(A)
    A_f=R_i@Q_i
    while(la.norm((np.diag(A)-np.diag(A_f)),2)>tol):
        A=A_f
        Q_i,R_i=la.qr(A)
        A_f=R_i@Q_i
    return(np.diag(A_f))
A=np.random.rand(5,5)# 5x5 matrix with random numbers between 0 and 1
print(qreigs(A,1e-5))
