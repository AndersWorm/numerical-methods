#the n'th term of the Taylor series expansion of sin(x)
# is f^(n)(pi/3) * (x - pi/3)^n / n!
# f^(n)(pi/3) = sin(pi/3 + n*pi/2) since sin(x)=cos(x-pi/2) and cos(x)=sin(x+pi/2)
import numpy as np
import math
import matplotlib.pyplot as plt
def sin_n_term(x,x_0,n):
    return(np.sin(x_0 + n * np.pi/2) * (x - x_0)**n / math.factorial(n))
n=20
x=np.linspace(-np.pi/2, 3*np.pi/2,20)
y=0
x_0=np.pi/3
for n in range(n):
    y+=sin_n_term(x,x_0,n)
plt.plot(x,y)
plt.show()

