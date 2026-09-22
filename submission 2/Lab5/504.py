import numpy as np
import matplotlib.pyplot as plt

def lagx(x,xp):
    l_j_data={} #store the l_j values with a dictionary, where I can use xj as the key and l_j as the value 
    for xj in xp:
        l_j=1 # since we are multiplying, we start with 1
        for xi in xp:
            if xi==xj: # skips these cases
                continue
            l_j*= (x-xi)/(xj-xi) #multiplyu the values
        l_j_data[xj]=l_j # end of each loop, we store the value of l_j in the dictionary with xj as the key
    return(l_j_data)
xp=np.array([-1,0,1,2])
x=np.linspace(-1,2,1000)
lagrange=lagx(x,xp)
plt.figure(figsize=(8, 5))
for xj, lj in lagrange.items(): # since dictionary has its key as the first value and the rest is the value, we can unpack it like this
    plt.plot(x, lj, lw=2, label=f"$x_j$ = {xj}")

plt.axhline(0, color="0.7", lw=0.8, zorder=0)
plt.axhline(1, color="0.7", lw=0.8, ls=":", zorder=0)
plt.plot(xp, np.zeros_like(xp), "k|", ms=12)  

plt.xlabel("x"); plt.ylabel(r"$\ell_j(x)$")
plt.legend(); plt.show()