# Lab 3, Questions 3 and 4 - golden search procedure, combined with the bracketing
# function from Q2 to find all minima of f (Q3, range [-10,10]) and g (Q4, range [1,5])
import numpy as np
phi=(1+np.sqrt(5))/2
def f(x):
    return x*np.cos(x)+x*np.sin(x/2)
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
    return(minimas)

def golden_search(f,a,b,tol):
    while(abs(b-a)>tol):
        c=b-(b-a)/phi
        d=a+(b-a)/phi
        if f(c)<f(d):
                b=d
        if f(d)<f(c):
                a=c
    return (a+b)/2

def minimum_finder(f, a, b, n, tol):
    brackets = minbracket(f, a, b, n)
    return[golden_search(f, x_min, x_max, tol) for x_min, x_max in brackets]
def g(x):
    return(x**2+x*np.sin(x**2))


tol=1e-5
a=-10
b=10
n=40
a_4=1
b_4=5
minima=minimum_finder(f,a,b,n,tol)
minima_4=minimum_finder(g,a_4,b_4,n,tol)
for x in minima:
        print(f"x_3 = {x:.6f}")
for x in minima_4:
        print(f"x_4 = {x:.6f}")
