from scipy import sparse as sp
from scipy.sparse import linalg as sla
import numpy as np
A=sp.random(5000,5000,density=0.1,format="csc") #sparse matrix where 10% of the numbers are different from 0
B=sp.random(5000,1,density=0.1,format="csc")
x=sla.spsolve(A,B) #sparse version of la.solve returns normal array
c=A.todense()@x - B.todense().flatten() #because spsolve makes a flat array
print("c=",np.linalg.norm(c,2))