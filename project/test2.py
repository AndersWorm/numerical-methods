import numpy as np
import matplotlib.pyplot as plt
from part2 import H_number, H_time
from part3 import Lindblad
from part4P import cost
from scipy.sparse import linalg as sla
from myfunctions import powell_search, gaussian, rk4solve
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
# This code will take a while to run, probably around a minute.

#create the hamilton and the initial density matrix
##
E_c = 2*np.pi * 0.25   # rad/ns is E_c/h 0.25 GHz 50x smaller than E_j
E_j = 2*np.pi * 12.5   # rad/ns # same as before is E_j/h 12.5 GHz
N=np.arange(-10,11,1) # the number of Cooper pairs from -10 to 10

t1, t2= 200, 150 # !!!!! the relaxation and dephasing times in ns these are the actual values of T1 and T2 that we will compare to the fitted values

gamma= 1.0 * np.array([1/t1,2 * (1/t2 - 1 / (2*t1))]) # the dissipation rates for the two levels
T1 = np.array([[0,1,0],[0,0,np.sqrt(2)],[0,0,0]],dtype=complex) # the lowering operator a on 3 levels
T2 = np.diag([0,1,2]) # adagger a 
T = np.array([T1,T2]) # the array of the two operators
T_g = 10 # gate time in ns used to the pi pulse
dT_g=0.005 # stepsize
T_R = 500 # gate time in ns for the rabi oscillation
T_final = 800 # the time after the gate time to see how the population of the first excited state decays
t_final = np.arange(T_g, T_final, dT_g)
rho0 = np.array([[1,0,0],[0,0,0],[0,0,0]],dtype=complex) # the initial density matrix
mu = T_g/2 # the amplitude is at its maximum at half of the gate time
sigma = T_g/6 # make sure it is close to zero in the start and end

H_num = H_number(N, E_c, E_j) # the hamiltonian in n basis
E,U= sla.eigsh(H_num, k=3, which='SA') # the eigenvalues and eigenvectors in n basis
H_0= np.diag(E,0)- E[0] * np.eye(3) # the diagonal matrix of the eigenvalues with the ground state energy subtracted so that the ground state energy is zero.
w = E[1]-E[0] # the energy gap between theground state and the first excited state
A= U.T @ np.diag(N) @ U #the coupling term between the different states

t_grid = np.arange(0, T_g + dT_g, dT_g) #grid of time points for the integration from 0 to T_g with step size dT_g
area = np.trapezoid(gaussian(t_grid, mu, sigma), t_grid) #integrate the gaussian function over the time grid to get the area under the curve
omega0 = np.pi / (abs(A[0, 1]) * area) # theta = C* integral of omega(t) dt the rotation angle has to be pi because that is the angle between ground state and first excited state in the bloch sphere representation. 
# this guess for omega0 is based on a two level system and does not take into account the third level. The optimal omega0 will be found using the powell_search function and considering the third level. 


# This will go through the last two parts of the project.
# From part 4 where I will find the optimal amplitude of the drive and the optimal drive frequency to achieve a pi pulse.
# I first did a 1D optimization where I only optimized the amplitude of the drive and then I did a 2D optimization where I optimized both the amplitude of the drive and the drive frequency.
# Im only gonna do the 2D optimization the code takes approximately 30 seconds to run on my computer.
x0 = np.array([omega0,w]) # the initial guess for omega0 and w
pset = np.eye(2) # the initial direction vectors are the unit vectors in 2D meaning I will first only change omega0 and then only change w
n, N_max = 10, 10 # the number of points and maximum number of sweeps to use in the minimum_finder function
tol = 1e-4 # the tolerance for the powell_search
T_p=5 # it will search in the radius T_p*dx_i
dOmega = 0.1 * omega0 # the step size for omega0
dw = 0.2/ sigma # the step size for w 1/sigma is approximately the width of the resonance valley in the cost function. So I will use 0.2/sigma as the step size for w to make sure I do not miss the minimum.
scale = np.array([dOmega, dw])
x_min, f_min, sweeps = powell_search(cost, x0, pset, N_max, n, T_p, tol, scale=scale)
print(f"The optimal omega0 is {x_min[0]/(2*np.pi):.6f} GHz and the optimal w is: is {x_min[1]/(2*np.pi):.6f} GHz which differs from qubit resonance without drive by: {x_min[1]/(2*np.pi)-x0[1]/(2*np.pi):.6f} GHz and the population of the first excited state at the end of the gate time is: {1-f_min:.6f}")
omega0_opt, w_opt = x_min
def f(t,rho): # the function that will be used in the RK4 method to solve the Lindblad equation
    omega = omega0_opt*gaussian(t,mu,sigma)
    H = H_time(t,H_0,A,omega,w_opt)
    return Lindblad(T, rho, H, gamma)
t_L,C_L = rk4solve(f, rho0, 0, dT_g, T_g)
plt.figure(figsize=(10,6))
plt.plot(t_L, C_L[:, 0, 0].real, label='|c0|^2')
plt.plot(t_L, C_L[:, 1, 1].real, label='|c1|^2')
plt.plot(t_L, C_L[:, 2, 2].real, label='|c2|^2')
plt.xlabel('Time (ns)')
plt.ylabel('Population')
plt.title('Optimal Gaussian pi Pulse for a Three-Level System with Lindblad Dissipation')
plt.legend()
plt.grid()
#From part 5 I will estimate the rabi frequency of the qubit and the two dissipation rates t1 from relaxation and t2 from dephasing. 

#estimation of the rabi frequency.
omega= E_c/20# the amplitude of the drive must be smaller than E_c  # the amplitude of the drive in rad/ns taken from part 4
def f(t,rho): 
    H = H_time(t,H_0,A,omega,w)
    return Lindblad(T, rho, H, gamma)
t_R,C_R = rk4solve(f, rho0, 0, dT_g, T_R)
P1_R = C_R[:, 1, 1].real # the population of the first excited state
peaks, _ = find_peaks(P1_R, prominence=0.2)            # index for peaks in the data
t_peaks, y_peaks = t_R[peaks], P1_R[peaks]
def model_U(t, gamma, rabi, b):
    return b - 0.5 * np.exp(-gamma * t) * np.cos(rabi * t) #The offset b is expected to be around 0.5, but it is left as a free parameter in the fit.
b_guess = 0.5
Omega_guess = 2 * np.pi / np.mean(np.diff(t_peaks)) # because the period of the oscillation is the time difference between the peaks, and the Rabi frequency is 2*pi/period. np.diff(t_peaks) gives the time difference between each pair of adjacent peaks.
gamma_guess = np.log((y_peaks[0] - b_guess) / (y_peaks[-1] - b_guess)) / (t_peaks[-1] - t_peaks[0])  # the decay rate is estimated from the logarithm of the ratio of the first and last peak heights, divided by the time difference between the first and last peaks.
p0 = [gamma_guess, Omega_guess, b_guess] # initial guess for the parameters
par, cov = curve_fit(model_U, t_R,P1_R, p0=p0) # fit the model to the data
rabi_frequency = par[1]
omega_fit = rabi_frequency / abs(A[0, 1]) # the amplitude of the drive in rad/ns

plt.figure(figsize=(10,6))
plt.plot(t_R, C_R[:, 0, 0].real, label='|c0|^2')
plt.plot(t_R, C_R[:, 1, 1].real, label='|c1|^2')
plt.plot(t_R, C_R[:, 2, 2].real, label='|c2|^2')
plt.plot(t_R, model_U(t_R, *par), label='Fit', color='red', linestyle='--')
plt.xlabel('Time (ns)')
plt.ylabel('Population')
plt.title('Three-Level System with Lindblad Dissipation fitted to determine Rabi frequency')
plt.legend()
plt.grid()
print(f"Fitted parameters: gamma = {par[0]:.4f}, Rabi frequency = {par[1]:.4f} , offset = {par[2]:.4f} ") 
print(f"actual omega0 is {omega / (2*np.pi):.6f}  GHz, fitted omega0 is {omega_fit/(2*np.pi):.6f} GHz which differs from actual omega0 by {abs(omega_fit/(2*np.pi)-omega/(2*np.pi)):.6f} GHz")
# Estimation of the dissipation rates t1 and t2.
#estimation of t1.
w= w_opt # the optimal w found in part 4.
def f(t,rho): # the function that will be used in the RK4 method to solve the Lindblad equation
    omega = omega0*gaussian(t,mu,sigma)
    H = H_time(t,H_0,A,omega,w)
    return Lindblad(T, rho, H, gamma)
t_L,C_L = rk4solve(f, rho0, 0, dT_g, T_g)
P1 = C_L[:, 1, 1].real
rho_final = C_L[-1] # the final density matrix after the gate time
def f(t,rho): # the function that will be used in the RK4 method to solve the Lindblad equation
    omega = omega0*gaussian(t,mu,sigma)
    H = H_time(t,H_0,A,omega,w)
    return Lindblad(T, rho, H, gamma)
t_Lf,C_Lf = rk4solve(f, rho0, 0, dT_g, T_final)

mask = t_Lf >= T_g
t_decay = t_Lf[mask]
P1_decay = C_Lf[:, 1, 1].real[mask]

def T1_fit(t, a, T1, b): # the function that will be used to fit the data to a decaying exponential function
    return a * np.exp(-t/T1) + b # b is because the population of the first excited state does not decay to zero.
A_guess = P1_decay[0] # the initial guess for the amplitude of the decaying exponential function
b_guess = 1 - A_guess # the initial guess for the offset of the decaying exponential function
P1_final = np.max(P1) * 1/np.exp(1) # the population of the first excited state after the gate time decays to 1/e of its maximum value
P1_f= abs(P1_decay- P1_final) # use absolute value so the minimum is the point closest to zero.
t_gamma = t_decay[np.argmin(P1_f)] - t_decay[0] # the time when the population of the first excited state is down to 1/e of its maximum value

p0 = [A_guess, t_gamma, b_guess] # initial guess for the parameters
par, cov = curve_fit(T1_fit, t_decay, P1_decay, p0=p0) # fit the model to the data

T1_fit_value = par[1] # the fitted value of T1

plt.figure(figsize=(10,6))
plt.plot(t_decay, P1_decay, label='Population of the first excited state after gate time', color='orange')
plt.plot(t_decay, T1_fit(t_decay, *par), label='Fitted decaying exponential function', color='green', linestyle='--')
plt.title('Population of the first excited state fitted to a decaying exponential function to determine T1', fontsize=16)
plt.xlabel('Time (ns)', fontsize=14)
plt.ylabel('Population', fontsize=14)   
plt.legend()
print(f'The fitted value of T1 is {T1_fit_value:.2f} ns the actual value of T1 is {t1:.2f} ns which differs from the fitted value by {abs(T1_fit_value-t1):.2f} ns')

# estimation of t2. 
w = E[1]-E[0]  # Taken from part 4.
def f(t,rho): 
    H = H_time(t,H_0,A,omega,w)
    return Lindblad(T, rho, H, gamma)
t_R,C_R = rk4solve(f, rho0, 0, dT_g, T_R) #use the longer gate time to see the dephasing of the qubit.
P0 = C_R[:, 0, 0].real # the population of the ground state

def T2_fit(t,b,T2, rabi):
    return b + 0.5 * np.exp(-t/T2) * np.cos(rabi * t) # since the P0= 1-P1
b_guess = 0.5 # the initial guess for the offset of the decaying exponential function
rabi_guess = 0.0850 # The rabi frequency founnd in part5_Rabi.py
peaks, _ = find_peaks(P0, prominence=0.1)            # index for peaks in the data
t_peaks = t_R[peaks]
y_peaks = P0[peaks]
peaks, _ = find_peaks(P0, prominence=0.05)            # index for peaks in the data
T2_guess = 1 / (np.log((P0[peaks[0]]-0.5)/(P0[peaks[1]]-0.5))/(t_R[peaks[1]]-t_R[peaks[0]])) # the initial guess for T2 using the first two peaks
t1_fit =200.83 # ns from part5_T1.py

p0 = [b_guess, T2_guess, rabi_guess] # the initial guess for the parameters
par, cov = curve_fit(T2_fit,t_R,P0,p0=p0) # fit the data to the decaying exponential function

t2_fit = par[1] * t1_fit / (2*t1_fit - par[1]) # t2_fit

plt.figure(figsize=(10,6))
plt.plot(t_R, P0, label='Data', color='blue')
plt.plot(t_R, T2_fit(t_R, *par), label='Fit', color='red', linestyle='--')
plt.xlabel('Time (ns)')
plt.ylabel('Population of Ground State P0')
plt.title('Fitting T2 Decay')
plt.legend()
plt.grid()
print(f"The fitted value of t2 is: {t2_fit:.2f} ns which differs from the actual value of t2: {t2:.2f} ns by: {t2_fit-t2:.2f} ns")

plt.show()