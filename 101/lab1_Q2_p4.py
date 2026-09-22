import numpy as np
import matplotlib.pyplot as plt
def forward(x,h):
    return(np.sin(1/(x+h))*(x+h)**2)
x=np.linspace(-1,1,100)
h=0.001
exact=2*x*np.sin(1/x)-np.cos(1/x)
plt.plot(x,forward(x,h))
plt.plot(x,exact)
plt.show()


