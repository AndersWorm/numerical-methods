import numpy as np
import matplotlib.pyplot as plt
N=1000
rng = np.random.default_rng()
p=rng.uniform(-1,1,size=(N,2)) #prints two numbers each time both between a,b and does it N times
x=p[:,0] #first column   
y=p[:,1] #second column 
plt.scatter(x,y)
plt.xlabel("x"); plt.ylabel("y")
plt.title("1000 uniformly points between x,y=[-1,1]")
plt.show()

def region(x, y):
    return x**2 + y**2 <= 1 #very simple because python will just check if the equation is true or false

inside = region(x, y)      # fx [True, False, True, ...]
x_in, y_in = x[inside], y[inside] #because numpy will only use the one with index true
plt.scatter(x_in,y_in)
plt.xlabel("x"); plt.ylabel("y")
plt.title("subset of random points inside the unit circle")
plt.show()