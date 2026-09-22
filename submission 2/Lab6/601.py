import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return(x**(3)+1)


def qtrapn(f,a,b,N,):
    h = (b-a)/N # interval steps
    I=0 #make sure it gives the same every time
    for i in range(N+1): #+1 to include the end point
        x=a+i*h
        if i==0 or i==(N): 
            I+=f(x) #make sure the end points are not counted double
        else:
            I+=2*f(x) # the rest are
    I = I*h/2
    return I


print("my trapezoidal function=",qtrapn(f,0,2,100))
x=np.linspace(0,2,100)
print("numpy's trapezoid function",np.trapezoid(f(x),x)) # for fun I compared it to numpys function
print("difference between them=", abs(qtrapn(f,0,2,100)-np.trapezoid(f(x),x)))
print("real value is 6.0 so both meassurements are good")

N=2# starting value for N 
I=[]# empty list to store the results
while (N<1024): #make sure it does not go over 1024
    N*=2 # double N each time
    I.append(qtrapn(f,0,2,N)) #integrate and append the result to the list
n=np.linspace(0,np.max(N),len(I)) #create a linspace for the x-axis with the same length as I
plt.scatter(n,I) 
plt.title("Trapezoidal integration of f(x)=x^3+1")
plt.xlabel("Number of intervals (N)")
plt.ylabel("Integral value")


def qsimpn(f,a,b,N):
    h = (b-a)/N
    I=0
    for i in range(N+1):
        x=a+i*h
        if i==0 or i==(N):
            I+=f(x)
        elif i%2==0:
            I+=2*f(x)
        else:
            I+=4*f(x)
    I = I*h/3
    return I
print("simpson function on f =",qsimpn(f,0,2,100)," with 100 iterations the others have 100 as well")

def errf(N_i,f,a,b):
    return(abs(qtrapn(f,a,b,N_i)-qtrapn(f,a,b,2*N_i)))
#calculates the difference between the two last points
def qtrapz(f,a,b,tol):
    N_i=1 #starting value for N
    iterations=0
    err=errf(N_i,f,a,b) 
    while err>tol:
        N_i*=2 #double N each time
        err=errf(N_i,f,a,b)
        iterations+=1
        if iterations>15: #just to avoid infinite loops
            print("Warning: Maximum number of iterations reached. The desired tolerance may not have been achieved.")
            break
    return("Number of intervals: {}, Result: {}, Tolerance: {}".format(N_i, qtrapn(f,a,b,N_i), tol))
def g(x):
    return(np.sin(x)**2 +np.exp(-x**2))
print("Trapezoidal function for g(x):", qtrapz(g,0,2,1e-8))
plt.show()