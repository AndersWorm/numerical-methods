import numpy as np
from pathlib import Path
import sys

module_folder = Path(
    "/Users/andersworm/Downloads/numerical methods/opgaver/102")
#path to the folder where the module is located

sys.path.insert(0, str(module_folder))  #tells python to look in the module folder for imports

from mysearch import minbracket
from mysearch import golden_search

from itertools import cycle

p_1=np.array([1,0]) # x unit vector
p_2=np.array([0,1]) # y unit vector
pset=np.array([p_1,p_2]) #creates an array with the two unit vectors
p_n=p_1 #initial direction is p_1
def f(x,y):
    return (x-1)**2 + y**2

def fline(t):
    x=x_n+t*p_n #gives me two values x=x[0] y=x[1]
    z=f(x[0],x[1]) # calculates f(x,y)
    return z

x_n=np.array([-2,-1]) #starting point
index=cycle([0,1]) # cycles between 0 and 1
tol=1e-5
tol_g=1e-7
T=4 # how far I go to each side of 0.
N=20 # number of lines how fine I bracket.
N_max = 10 #maximum number of iterations I want
z_i=fline(0) # for calculating the err
n_dir=len(pset) #tells me how many iterations i have to do before it is a sweep
#The result is only valid if it is a full sweep meaning i have done x and y
# so i number of iterations must have (i+1)% 2=0 since i starts on 0. So i=1 is a full sweep

for i in range(N_max):
    p_n=pset[next(index)] #shifts between p_x and p_y for each iteration
    brackets = minbracket(fline, -T,T, N) # minbrackets uses fline and run through -t to t in N steps
    if not brackets: #means if brackets are empty
        print("no minima found")
        break
    a,b=brackets[0] #takes the first bracket and assigns it to a and b
    t_min=golden_search(fline,a, b,tol_g)
    x_n=x_n+t_min*p_n #my new set of coordinates
    z_f=fline(0) #but now it uses x_n calculated above so not the same as z_i
    if (i+1)%2==0:
        if abs(z_f-z_i)<tol:
            print(f"minima at x={x_n}, f={fline(t_min)}", "after",(i+1)//n_dir ,"sweeps:")
            break
        z_i=z_f #updates it and start over, i only updates after a full sweep so it is valid to compare the two values.
    if i==N_max-1:
        print("Reached maximum number of iterations")
