import numpy as np
from scipy import linalg as la
A=np.random.rand(20,20)# 20x20 matrix with random numbers between 0 and 1
B=np.random.rand(20,1) #20x1 matrix
#by finding A^-1 i can solve x as A^-1@b
x=la.inv(A)@B
print("Ax-B=",abs(np.round(A@x-B,10)))
#I need to use abs and round because. the inverse are not perfect I will get some errors by the scale of e-17
# and -0 which are just 0