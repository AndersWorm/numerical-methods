from scipy import sparse as sp
from scipy.sparse import linalg as sla
import matplotlib.pyplot as plt
A = sp.random(500, 500, density=0.1, format="csc")
lu = sla.splu(A) #the data from lu ar stored in a super object
AL=lu.L
AU=lu.U