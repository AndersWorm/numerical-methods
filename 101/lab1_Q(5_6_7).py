import numpy as np
import matplotlib.pyplot as plt
def plot(a,b): #define a function to plot a and b
    plt.plot(a,b)
    return( plt.show())

x=np.linspace(-2,2,100)
a=x**2

y=np.linspace(-10,10,100)
b= np.sin(2*y) / y

o=np.linspace(0,3,100)
c=np.exp(2*o)

k=np.linspace(0,4,100)
d=4**(2*k)

g=np.linspace(-10,10,100)
j= np.exp(2*g)
plot(x,a)
plot(y,b)
plot(o,c)
plot(k,d)
plt.semilogy(g,j)
plt.show()
q=np.logspace(0.001,10,100)
u=4**(2*q)
plt.loglog(q,u)
plt.show()
q_=np.linspace(0.001,10,100)
u_=4**(2*q_)
plt.plot(q_,u_)
plt.show()

