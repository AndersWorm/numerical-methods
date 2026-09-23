from scipy import sparse as sp
from scipy.sparse import linalg as sla
import numpy as np
import matplotlib.pyplot as plt
A=(np.random.randn(20,20)-0.5)**5 # randn is the normal distribution with mean value zero and standard deviation 1 unlike rand which is uniformly values between 0 and 1
# ^5 works on every index and -1/2 subtracts from every index
def zerof(A,m):
    B=A.copy()
    B[abs(A)<m]=0 #abs(A) runs through the entire matrix and compares them to m
    #and returns a true and false array which A[] uses and replace all the true with 0
    return(B)
A2=zerof(A,0.1)
plt.spy(A2)
plt.show() #white means the value is zero black means different from zero

#the numbers stored are float(64) meaning one number is 64 bits which is 8 bytes
#so the size of a sparse matrix is its size how many numbers it have times 8
B=(np.random.randn(5000,5000)-0.5)**5 # randn is the normal distribution with mean value zero and standard deviation 1 unlike rand which is uniformly values between 0 and 1
# ^5 works on every index and -1/2 subtracts from every index
B2=sp.csr_matrix(zerof(B,0.1))
SD=B.nbytes*1e-6 #convert to megabyte
SS=B2.size*8 *1e-6
print("difference in size SD-SS=",SD-SS,"MB")