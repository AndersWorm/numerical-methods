# Lab 2, Question 4b - secant method to find all roots of f(x) = x^4 - x - 1
# (moved over from the 102/202.ipynb notebook)
import numpy as np

a=-2
b=2
n=100
a_s=-10
b_s=10

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
            return(np.array(brackets))

# Starts in A, B makes the secant between A and B, C is where the secant intersects the x-axis.
# Now the two new points are B and C, we make the secant between them and find D. Then the
# new points are C and D, we make the secant between them and find E. This continues until we
# have found a root with the desired accuracy.
def sechant_meth(x_1, x_2,f):
    return(x_1-f(x_1)*(x_1-x_2)/(f(x_1)-f(x_2)))
def secant_root(f, x_1, x_2, tol=1e-10, maxit=100):
    for i in range(maxit):
        x = sechant_meth(x_1, x_2, f)
        if abs(x - x_2) < tol:
            return x
        x_1, x_2 = x_2, x
    return None                       # reached maxit without converging

brackets = brack(f,a,b,n)
roots = []

for x_1, x_2 in brackets:            # unpacks [-0.76, -0.72], so [1.2, 1.24]
    r = secant_root(f, x_1, x_2)
    if r is None:
        print("Did not converge in [", x_1, ",", x_2, "]")
    else:
        roots.append(r)
        print("Root found at x =", r)

roots = np.array(roots)
