import numpy as np
from scipy import linalg as la
#I use the concept of the previous question where A is all the numbers infront of F, x is a coloumn matrix with the F's and B is the result
A=np.array([-1,-3,2,3,-6,2,0,2,-1]).reshape(-1,3)
B=np.array([4,3,2]).reshape(-1,1)
print(A)
x=la.inv(A)@B
print("The solution of the equation x=",x)
print("confirmation difference between A@x and B =",A@x-B)