from scipy import sparse as sp
from scipy.sparse import linalg as sla
import matplotlib.pyplot as plt
A = sp.random(500, 500, density=0.1, format="csc")
lu = sla.splu(A) #the data from lu ar stored in a super object
plt.spy(lu.L, markersize=1) #lu.L calls the lower data 
plt.title("L")
plt.show()

plt.spy(lu.U, markersize=1) #calls the upper data
plt.title("U") 
plt.show()