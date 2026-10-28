import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import linalg as sla
from scipy import sparse as sp
from myfunctions import rk4solve
#I use the number basis this time because it is easier to work with
def H_number(N, Ec, Ej):
    M=len(N)-1 # since the upper and lower diagonal has 1 less element than the diagonal
    K= 4*Ec*np.diag(N**2,0)
    V= -Ej/2 * (np.diag(np.ones(M),1)+np.diag(np.ones(M),-1))
    H=sp.csr_matrix(K+V)
    return(H) 
def H_drive(t,omega, w,A):
    return(omega*np.sin(w*t)*A) 

def H_time(t,H0,A,omega,w): #the time dependent hamiltonian
    return(H0+H_drive(t,omega,w,A))

def f(t,c): #the function for the ODE solver
    return(-1j*H_time(t,H_0,A,omega,w)@c)

if __name__ == "__main__": # so that the code below only runs when this file is run directly and not when it is imported into another file.
    #
    N=np.arange(-10,11,1) # the number of Cooper pairs from -10 to 10
    E_c = 2*np.pi * 0.25   # rad/ns is E_c/h 0.25 GHz
    E_j = 2*np.pi * 12.5   # rad/ns # same as before is E_j/h 12.5 GHz
    omega= E_c/3 # the amplitude of the drive must be smaller than E_c
    H_num = H_number(N, E_c, E_j) # the hamiltonian in n basis
    E,U= sla.eigsh(H_num, k=3, which='SA') # the eigenvalues and eigenvectors in n basis
    H_0= np.diag(E,0)- E[0] * np.eye(3) # the diagonal matrix of the eigenvalues with the ground state energy subtracted so that the ground state energy is zero.
    w = E[1]-E[0] # the energy gap between the ground state and the first excited state

    A= U.T @ np.diag(N) @ U #the coupling term between the different states

    #
    # Now have a ODE where dy/dt = f(t,y) which I can solve using RK4 from computer lab.


    # RK4 needs the function, the initial condition, the initial time, the step size, and the maximum time. 
    # The system will start in the ground state c0=[1,0,0] and I will use a step size of 0.005 and a maximum time of 5000.
    c0=np.array([1,0,0]) # the initial condition
    t,C = rk4solve(f, c0, 0, 0.005, 25)

    plt.plot(t, np.abs(C[:, 0])**2, label='|c0|^2')
    plt.plot(t, np.abs(C[:, 1])**2, label='|c1|^2')
    plt.plot(t, np.abs(C[:, 2])**2, label='|c2|^2')
    plt.xlabel('Time (ns)')
    plt.ylabel('Probability')
    plt.title('Time Evolution of State Probabilities')
    plt.legend(loc='upper right')
    plt.show()
