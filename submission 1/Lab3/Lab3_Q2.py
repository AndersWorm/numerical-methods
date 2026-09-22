# Lab 3, Question 2 - minbracket function, plus printing the list of bracketing intervals
import numpy as np
def f(x):
    return x*np.cos(x)+x*np.sin(x/2)
a=-5
b=5
n=20
def minbracket(f,a,b,n):
    minimas=[]
    dx=abs(a-b)/n
    x_1=a
    c=a+dx
    x_2=a+2*dx
    while (x_2<= b):
        if f(x_1)>f(c) and f(x_2)>f(c):
            minimas.append([x_1,x_2])
            x_1+=2*dx
            c+=2*dx
            x_2+=2*dx
        else:
            x_1+=dx
            c+=dx
            x_2+=dx
    return(minimas,print(minimas))
blist=minbracket(f,a,b,n)
