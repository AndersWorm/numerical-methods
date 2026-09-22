import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
def region(x, y):
    return x**2 + y**2 <= 1

def monty2d(f,region,a,b,M):
    I=0
    V=abs(b-a)**2 #the volume of a 2d function is the area
    sampler=qmc.Sobol(d=2) #initialize the generator in 2d so i get x and y each time
    N=2**M
    p=sampler.random_base2(M)
    x,y =(b-a)*p[:,0]+a,(b-a)*p[:,1]+a
    inside=region(x,y)
    I=np.sum(f( x[inside],y[inside]))
    return(I*V/N)
f = lambda x, y: np.ones_like(x)    # f = 1
print(monty2d(f, region, -1, 1, 10))   # should return 3.14