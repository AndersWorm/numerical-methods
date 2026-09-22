# Lab 2, Question 3 - bisection routine, tested via mysearch.py (see mysearch.py in this folder)
import numpy as np
import mysearch as mys
from mysearch import bisection
def f(x):
    f =  x-np.cos(x)
    return(f)
a=0
b=1
tol=0.001
x=mys.bisection(f,a,b,tol)
print("Zero found at x =",x)
