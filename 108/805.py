import numpy as np
from scipy import linalg as la
A=np.random.rand(4,4)# 4x4 matrix with random numbers between 0 and 1
P,L,U=la.lu(A)
print("lower must only have number at the diagonal or below and only ones on the diagonal",L)
print("upper must only have numbers at the diagonal or above",U)
Q,R=la.qr(A)
Qin=la.inv(Q)
print("QR decomposition of A R=",R) #must be an upper matrix
print("norm between Q.T and Qinvgit",np.round(la.norm(Q.T - la.inv(Q), 2),10))