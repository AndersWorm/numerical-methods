import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse as sp
from scipy.sparse import linalg as sla
# Transmon hamiltonian is given by H = -4Ec d^2/dphi^2 - Ej cos(phi)
# use finite difference method to approximate the second derivative and use sparse matrices to save memory and computation time.
def SD2(N,h): #use the definition from the lecture notes.
    M=N-1 # since the upper and lower diagonal has 1 less element than the diagonal
    AD=np.diag(-2*np.ones(N),0)# 0 main diagonal
    AO=np.diag(np.ones(M),1)# +1 over main diagonal
    AL=np.diag(np.ones(M),-1)# -1 over main diagonal
    AP = np.diag([1],M) + np.diag([1],-M) # the boundary condition because it is periodic.
    D2=sp.csr_matrix((AD+AO+AL+AP)/h**2)
    return(D2)
# The next turn in the hamilton -Ej cos(phi) is a diagonal matrix with the values of -Ej cos(phi) on the diagonal.
def SV(phi):
    AD=np.diag(np.cos(phi),0)# 0 main diagonal
    SV=sp.csr_matrix(AD)
    return(SV)

# Make the full hamiltonian by combining the two parts.
def transmon_H(N,h,E_j,E_c,phi):
    H = -4*E_c *SD2(N,h)- E_j*SV(phi)
    return(H)
