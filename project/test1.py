# This test will go through the first three parts of the projekt.
from Part1 import transmon_H
from part2 import H_number, H_drive, H_time
from part3 import Dissipation, Lindblad
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import linalg as sla
from scipy import sparse as sp
from myfunctions import rk4solve

#starting values
N_1=1000 # number of points in the phi grid
h= 2*np.pi /N_1 # the step size in phi
E_c = 2*np.pi * 0.25   # rad/ns is E_c/h 0.25 GHz
E_j = 2*np.pi * 12.5   # rad/ns so E_j is 50x larger than E_c as you want in a transmon qubit
phi = np.linspace(- np.pi, np.pi, N_1, endpoint=False) 
c0=np.array([1,0,0]) # the initial condition
N=np.arange(-10,11,1) # the number of Cooper pairs from -10 to 10
omega= E_c/3 # the amplitude of the drive must be smaller than E_c
t1, t2= 200, 150 # the relaxation and dephasing times in ns
gamma=np.array([1/t1,2 * (1/t2 - 1 / (2*t1))]) # the dissipation rates for the two levels
T1 = np.array([[0,1,0],[0,0,np.sqrt(2)],[0,0,0]],dtype=complex) # the lowering operator a on 3 levels
T2 = np.diag([0,1,2]) # adagger a 
T = np.array([T1,T2]) # the array of the two operators
rho0 = np.array([[1,0,0],[0,0,0],[0,0,0]],dtype=complex) # the initial density matrix


# from part 1 plot the wavefunctions of the first three states and print the eigenvalues and the energy gaps between the states.
H=transmon_H(N_1, h, E_j, E_c, phi)
E, U = sla.eigsh(H, k=3, which='SA') # I use the sparse matrix to find the eigenvalues and eigenvectors. k is the number of eigenvalues and eigenvectors to find. which='SA' means that I want the smallest algebraic eigenvalues.
plt.figure(figsize=(10,6))
plt.plot(phi/(np.pi),U[:,0], label='Ground state') # the ground state wavefunction
plt.plot(phi/(np.pi),U[:,1], label='First excited state') # the first excited state wavefunction
plt.plot(phi/(np.pi),U[:,2], label='Second excited state') # the second excited state wavefunction
plt.legend()
plt.xlabel('$\phi$ / $\pi$')
plt.ylabel('Wavefunction')
plt.title('Wavefunctions of the first three states')
plt.grid()
print('Eigenvalues:', 'E0=', E[0], 'E1=', E[1], 'E2=', E[2]) # the eigenvalues of the hamiltonian
print('Energy gap between ground state and first excited state:', 'DE_01=', E[1]-E[0]) 
print('Energy gap between first excited state and second excited state:', 'DE_12=', E[2]-E[1]) 
print('difference between the energy gaps:', 'DE_12-DE_01=', (E[2]-E[1])-(E[1]-E[0]))


# from part 2 plot the time evolution of the state probabilities for the first three states using the RK4 method to solve the ODE.
H_num = H_number(N, E_c, E_j) # the hamiltonian in n basis
E,U= sla.eigsh(H_num, k=3, which='SA') # the eigenvalues and eigenvectors in n basis
H_0= np.diag(E,0)- E[0] * np.eye(3) # the diagonal matrix of the eigenvalues with the ground state energy subtracted so that the ground state energy is zero.
w = E[1]-E[0] # the energy gap between the ground state and the first excited state
A= U.T @ np.diag(N) @ U #the coupling term between the different states
def f(t,c): #the function for the ODE solver
    return(-1j*H_time(t,H_0,A,omega,w)@c)
t,C = rk4solve(f, c0, 0, 0.005, 25)

plt.figure(figsize=(10,6))
plt.plot(t, np.abs(C[:, 0])**2, label='|c0|^2')
plt.plot(t, np.abs(C[:, 1])**2, label='|c1|^2')
plt.plot(t, np.abs(C[:, 2])**2, label='|c2|^2')
plt.xlabel('Time (ns)')
plt.ylabel('Probability')
plt.title('Population Dynamics of a Three-Level System without Lindblad Dissipation')
plt.legend(loc='upper right')
plt.grid()

#from part 3 now it is extended to Lindblad equation
def f(t,rho): # the function that will be used in the RK4 method to solve the Lindblad equation
    H = H_time(t,H_0,A,omega,w)
    return Lindblad(T, rho, H, gamma)

t, R = rk4solve(f, rho0, 0, 0.005, 25)
plt.figure(figsize=(10, 6))
plt.plot(t, R[:, 0, 0].real, label='|c0|^2')
plt.plot(t, R[:, 1, 1].real, label='|c1|^2')
plt.plot(t, R[:, 2, 2].real, label='|c2|^2')
plt.xlabel('Time (ns)')
plt.ylabel('Population')
plt.title('Population Dynamics of a Three-Level System with Lindblad Dissipation')
plt.legend()
plt.grid()

plt.show()
