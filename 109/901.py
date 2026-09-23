from scipy import sparse as sp
from scipy.sparse import linalg as sla
import numpy as np
AD=np.diag([1, 2, 3, 4, 5],0)# 0 main diagonal
AS=sp.csr_matrix(AD)
print("AS=",AS)
print("A=",AS.todense()) # to.array() converts them back to arrays
BD=AD+np.diag([1, 1, 1, 1],1)# 1 above main diagonal
BS =sp.csr_matrix(BD) # so the first array goes in main 0 and the second array goes in 1 because it is the first diagonal above the main diagonal
print("BS=",BS)
print("B=",BS.todense())

#faster way use
AS = sp.diags([1,2,3,4,5], 0, format="csr")
BS = sp.diags([[1,2,3,4,5], [1,1,1,1]], [0,1], format="csr")
