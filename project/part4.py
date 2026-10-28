import numpy as np
import matplotlib.pyplot as plt
from part2 import H_number, H_time
from scipy.sparse import linalg as sla
from myfunctions import rk4solve, lagint

#starting parameters
N=np.arange(-10,11,1) # the number of Cooper pairs from -10 to 10
E_c = 2*np.pi * 0.25   # rad/ns is E_c/h 0.25 GHz
E_j = 2*np.pi * 12.5   # rad/ns # same as before is E_j/h 12.5 GHz
omega= E_c/3 # the amplitude of the drive must be smaller than E_c
H_num = H_number(N, E_c, E_j) # the hamiltonian in n basis
E,U= sla.eigsh(H_num, k=3, which='SA') # the eigenvalues and eigenvectors in n basis
H_0= np.diag(E,0)- E[0] * np.eye(3) # the diagonal matrix of the eigenvalues with the ground state energy subtracted so that the ground state energy is zero.
w = E[1]-E[0] # the energy gap between the ground state and the first excited state
A= U.T @ np.diag(N) @ U #the coupling term between the different states

def f(t,c): #the function for the ODE solver
    return(-1j*H_time(t,H_0,A,omega,w)@c)
##

c0=np.array([1,0,0]) # the initial condition
t,C = rk4solve(f, c0, 0, 0.005, 12) #only the first 12 ns are needed to find the maximum of P1
P1 = np.abs(C[:, 1])**2
i_max = np.argmax(np.abs(C[:, 1])**2) #find the index of the maximum of P1
k=4 #number of points to use for interpolation
xp = t[i_max-2*k : i_max+2*k+1 : k] #interpolate around the maximum using 5 points.
yp = P1[i_max-2*k : i_max+2*k+1 : k]
x=np.linspace(xp[0], xp[-1], 1000) # create an array of x values for interpolation
i_max_interp = np.argmax(lagint(x,xp,yp))
t_max_interp = x[i_max_interp]
t_max = t[i_max]
plt.figure(figsize=(8, 5))
plt.plot(x, lagint(x,xp,yp), lw=2, label="Lagrange interpolation")
plt.plot(xp, yp, "ro", label="Data points")
plt.axvline(t_max_interp, color='g', linestyle='--', label=f'Max at t_int={t_max_interp:.4f} ns')   
plt.axvline(t_max, color='b', linestyle='--', label=f'Max at t={t_max:.4f} ns')
plt.xlabel("x"); plt.ylabel("L(x)")
plt.ylim(0.92, 0.93)
plt.legend(loc='upper right') #right corner
print(f'A pi pulse is achieved at t={t_max_interp:.4f} ns with a maximum population of P1={lagint(t_max_interp,xp,yp):.4f}')
plt.show()
