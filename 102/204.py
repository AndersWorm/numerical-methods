import numpy as np
def f(x):
    return(x**4 - x - 1)
def g(x):
    return(np.sin(x))
def sign(f):
    return(np.sign(f))

def brack(f,a,b,n):
    brackets=[]
    d= np.abs(b-a)/n
    x_1=a
    x_2=a+d
    while x_2<=b:
        s = sign(f(x_1))+sign(f(x_2))
        if s==0:
            brackets.append([x_1,x_2])
            x_1=x_2
            x_2+=d
        else:
            x_1=x_2
            x_2+=d
        if x_2>b:
            return(print(np.array(brackets)))

a=-2
b=2
n=100
a_s=-10
b_s=10
brack(f,a,b,n)
brack(g,a_s,b_s,n)