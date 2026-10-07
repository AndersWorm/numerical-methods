import numpy as np
from scipy import linalg as la
a=np.array([1,0,2,3])
b=np.array([1,3,-1,2])
A=a.reshape(1,-1) #not necessary because it is already on the right form but reshape(1,-1) means 1 row and as many columns needed to sufice that
B=b.reshape(-1,1) # as many rows needed but one column
print(A.shape, B.shape)
print(A,B)
print("dot product using @=",A@B)
print("dot product using np.dot=",np.dot(A,B))
print("Transpose of A=",A.T) #transpose
print("b abs squared=",B.T@B)
print("L2 norm of a=",la.norm(a,2))
