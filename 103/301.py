import numpy as np

def f(x):
    return x*np.cos(x)+x*np.sin(x/2)

x_min, x_max, N = -10, 10, 100
x=np.linspace(x_min,x_max,N)
dx=abs(x_min-x_max)/N

minimas=[]
x_1=x_min
c=x_1+dx
x_2=x_1+2*dx
while (x_2<= x_max):
    if f(x_1)>f(c) and f(x_2)>f(c):
        minimas.append([x_1,x_2])
        x_1+=2*dx
        c+=2*dx
        x_2+=2*dx
    else:
        x_1+=dx
        c+=dx
        x_2+=dx
