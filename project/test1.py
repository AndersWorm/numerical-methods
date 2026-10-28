from Part1 import transmon_H
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import linalg as sla
#starting values
N=1000
h= 2*np.pi /N
E_c = 2*np.pi * 0.25   # rad/ns is E_c/h 0.25 GHz
E_j = 2*np.pi * 12.5   # rad/ns # same as before is E_j/h 12.5 GHz
phi = np.linspace(- np.pi, np.pi, N, endpoint=False) 
H=transmon_H(N, h, E_j, E_c, phi)
E, U = sla.eigsh(H, k=3, which='SA') # I use the sparse matrix to find the eigenvalues and eigenvectors. k is the number of eigenvalues and eigenvectors to find. which='SA' means that I want the smallest algebraic eigenvalues.
plt.plot(phi/(np.pi),U[:,0], label='Ground state') # the ground state wavefunction
plt.plot(phi/(np.pi),U[:,1], label='First excited state') # the first excited state wavefunction
plt.plot(phi/(np.pi),U[:,2], label='Second excited state') # the second excited state wavefunction
plt.legend()
plt.xlabel('$\phi$ / $\pi$')
plt.ylabel('Wavefunction')
plt.title('Wavefunctions of the first three states')
print('Eigenvalues:', 'E0=', E[0], 'E1=', E[1], 'E2=', E[2]) # the eigenvalues of the hamiltonian
print('Energy gap between ground state and first excited state:', 'DE_01=', E[1]-E[0]) 
print('Energy gap between first excited state and second excited state:', 'DE_12=', E[2]-E[1]) 
print('difference between the energy gaps:', 'DE_12-DE_01=', (E[2]-E[1])-(E[1]-E[0]))
plt.show()
