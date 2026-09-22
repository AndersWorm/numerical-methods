import numpy as np
import matplotlib.pyplot as plt
def forward(x,h):
    return(( (np.sin(x+h))**2 - (np.sin(x))**2 ) / h)
def exact(x):
    return(2*np.sin(x)*np.cos(x)) # the true derivative of sin^2(x)
x=np.linspace(-10,10,10)
h=0.10
g=exact(x)
#print(forward(x,h))

def abs_err(f,g):
    return((np.abs(f-g)))


def frac_err(f,g):
    x = np.abs( (f-g) / (g) )
    return(x)

x_0=np.ones(100)
h_=np.logspace(-15,0,100)
f_h=forward(x_0,h_)
g_=exact(x_0)
abserr=abs_err(f_h,g_)
frac=frac_err(f_h,g_)

plt.loglog(h_,frac)
plt.title('Fractional error of forward difference approximation')
plt.xlabel('h')
plt.ylabel('Fractional error')


plt.loglog(h_,abserr)
plt.title('Absolute error of forward difference approximation')
plt.xlabel('h')
plt.ylabel('Absolute error')

opt=np.argmin(frac)
opt_h=h_[opt]
print(opt_h)
plt.show()
