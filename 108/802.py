import numpy as np
from scipy import linalg as la
A=np.random.rand(4,4)# 4x4 matrix with random numbers between 0 and 1
b=np.array([1,3,-1,2]).reshape(-1,1)
print(A)
print("Transpose of A=",A.T)
print("Dot product of A and b",A@b)
print("Inverse of A",la.inv(A))
I=abs(np.round(A@la.inv(A),15))
print("confirm that A^-1 @A=I", abs(np.round(A@la.inv(A),10)))
print("L2 norm difference between A^-1@A and I 4x4=",la.norm(I-np.eye(4),2))