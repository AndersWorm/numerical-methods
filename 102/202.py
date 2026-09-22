import numpy as np
def newton(x_n,f,df):
    return(x_n - f/df)
def err_c(x,a):
    return(np.abs(x-a))
def f(x):
    return(x-np.cos(x))
def df(x):
    return(1+np.sin(x))
a=0.7390851332
x_0=0.5
n=3
for n in range(n):
    x_n=newton(x_0,f(x_0),df(x_0))
    err=err_c(x_n,a)
    if err>0.00001:
        x_0=x_n
    else:
        print(err)
        print(x_n)
        print(n)
        break

