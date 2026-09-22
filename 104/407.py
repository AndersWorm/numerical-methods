from pathlib import Path
import sys
import numpy as np

module_folder = Path(
    "/Users/andersworm/Downloads/numerical methods/opgaver/102")
#path to the folder where the module is located

sys.path.insert(0, str(module_folder))  #tells python to look in the module folder for imports

from mysearch import powell_search

x_1, x_2 , x_3=np.array([-2.0,-1.0]), np.array([0.0,0.0]), np.array([2.0,1.0]) #starting points
pset=np.array([[1.0, 0.0], [0.0, 1.0]]) #creates an array with the two unit vectors
N_max = 10 #maximum number of iterations I want
N=20 #number of points I want to use for the minbracket function
T=4 # how far I go to each side of 0.
tol=1e-6 # tolerance for convergence
def f(x,y):
        return (x-1)**2 + y**2
for x_n in [x_1, x_2, x_3]:
    x_min, f_min, sweeps = powell_search(f, x_n, pset, N_max, N, T, tol)
    print("Starting point:", x_n)
    print("Minimum point:", x_min)
    print("Function value at minimum:", f_min)
    print("Number of sweeps:", sweeps)
    print()