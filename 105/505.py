import numpy as np
import matplotlib.pyplot as plt

def lagx(x,xp):
    l_j_data={}
    for xj in xp:
        l_j=1
        for xi in xp:
            if xi==xj:
                continue
            l_j*= (x-xi)/(xj-xi)
        l_j_data[xj]=l_j
    return(l_j_data)
def lagint(x,xp,yp):
    L=0
    lag=lagx(x,xp) #get all my lagrange functions
    for j, (xj, lj) in enumerate(lag.items()): # enumerate gives me the index j and the xj,lj values
        L += lj * yp[j] # use the index j to get the corresponding y value from yp
    return(L)
#test data if it works, meaning if the functions hits the points in xp,yp
xp=np.array([-1,0,1,2])
yp=np.array([2,-1,1,0])
x=np.linspace(-1,2,1000)
plt.figure(figsize=(8, 5))
plt.plot(x, lagint(x,xp,yp), lw=2, label="Lagrange interpolation")
plt.plot(xp, yp, "ro", label="Data points")
plt.axhline(0, color="0.7", lw=0.8, zorder=0)
plt.axhline(1, color="0.7", lw=0.8, ls=":", zorder=0)
plt.xlabel("x"); plt.ylabel("L(x)")
plt.legend(); plt.show()