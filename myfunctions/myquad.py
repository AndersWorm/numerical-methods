import numpy as np
import scipy as sp
from scipy.special import p_roots

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
def qsimpn(f,a,b,N):
    h = (b-a)/N
    I=0
    for i in range(N+1):
        x=a+i*h
        if i==0 or i==(N): #the end points
            I+=f(x)
        elif i%2==0: #all even points are multiplied by two
            I+=2*f(x)
        else:
            I+=4*f(x)   #all uneven points
    I = I*h/3
    return I

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


def quads_1(a,b,n,f):
    x,w= p_roots(n)  # x and w are the roots and weights of the Legendre polynomial of degree n
    I=0
    for i in range(n):
        x_tr=((b-a)/2)*x[i] + (b+a)/2 #scale the roots to the interval [a,b]
        I+=w[i]*f(x_tr) #sum the weighted function values at the scaled roots
    return(I*0.5*(b-a)) #use the formula from the slides

def squad2d(f,a,b,c,d,tol=1e-5):
    def yint(x): #I define my y integral which quadz will use for different values of x in the end
        def fy(y): #I define my dummy function fy which is f(x,y) but only y varies
            return(f(x,y))
        Iy,n=quadz(fy,c,d,tol) #,n becausse my quadz also returns n
        return(Iy)
    I,n=quadz(yint,a,b,tol)
    return(I)
#the code calculates yint for different values of x given by quadz and then quadz integrate that over x

    



def quadz(f, a, b, tol, n_max=200):
    n = 1
    I_i = quads_1(a, b, n, f)
    I_f = quads_1(a, b, n+1, f)
    while abs(I_i - I_f) > tol:
        n += 1
        if n >= n_max:
            raise RuntimeError(f"didnot reach={tol} with n={n_max}")
        I_i = I_f
        I_f = quads_1(a, b, n+1, f)
    return I_f, n+1

def region(x, y):
    return x**2 + y**2 <= 1

def monty2d(f,region,a,b,M):
    I=0
    V=abs(b-a)**2 #the volume of a 2d function is the area
    sampler=qmc.Sobol(d=2) #initialize the generator in 2d so i get x and y each time
    N=2**M
    p=sampler.random_base2(M)
    x,y =(b-a)*p[:,0]+a,(b-a)*p[:,1]+a
    inside=region(x,y)
    I=np.sum(f( x[inside],y[inside]))
    return(I*V/N)