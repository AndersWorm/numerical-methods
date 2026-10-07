import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
M=10 # I get 2^M points
sampler=qmc.Sobol(d=2) #initialize the generator in 2d so i get x and y each time
p=sampler.random_base2(M) # creates 2^M sobolt points between 0 and 1
p=2*p-1 # so it is from -1 to 1
x,y=p[:,0],p[:,1]

def region(x, y):
    return x**2 + y**2 <= 1 #very simple because python will just check if the equation is true or false

inside = region(x, y)      # fx [True, False, True, ...]
x_in, y_in = x[inside], y[inside] #because numpy will only use the one with index true
plt.scatter(x_in,y_in)
plt.xlabel("x"); plt.ylabel("y")
plt.title("subset of random sobolt points inside the unit circle")
plt.show()