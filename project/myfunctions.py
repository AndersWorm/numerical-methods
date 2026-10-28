import numpy as np
def rk4solve(f,y0,x0,h,xmax):
    ycoor=[y0] #starting value of y
    xcoor=[x0] #starting value of x
    while (xcoor[-1]<=xmax-h):
        x,y=xcoor[-1], ycoor[-1] # get the last values of x and y
        k1=f(x,y) #calculate k1 using the function f(x,y)
        k2=f( x + 1/2 *h, y + h/2 *k1) #calculate k2 using the function f(x,y) and the values of x and y
        k3=f(x+h/2,y+h/2 * k2) #calculate k3 
        k4=f(x+h,y+k3*h) #calculate k4 there was missing a h in the last term on k3 on the slides
        y=y+h/6*(k1+2*k2+2*k3+k4) # update y using the Runge-Kutta method
        xcoor.append(x+h),ycoor.append(y) # each element of the list ycoor becomes a row in the array Y, so Y will be a 2D array with each row corresponding to a time step and each column corresponding to a variable in the system of ODEs.
    return(np.array(xcoor),np.array(ycoor)) #return the x and y coordinates as numpy arrays 
def lagx(x,xp):
    l_j_data={}
    for xj in xp:
        l_j=1
        for xi in xp:
            if xi==xj:
                continue
            l_j*= (x-xi)/(xj-xi)
        l_j_data[xj]=l_j
    return(l_j_data)
def lagint(x,xp,yp):
    L=0
    lag=lagx(x,xp) #get all my lagrange functions
    for j, (xj, lj) in enumerate(lag.items()): # enumerate gives me the index j and the xj,lj values
        L += lj * yp[j] # use the index j to get the corresponding y value from yp
    return(L)

def sign(f):
    return(np.sign(f))
def err_f(a,b):
    return(np.abs(b-a))
def bisection(f,a,b,tol):
    n=0
    while err_f(a,b)>tol:
        s=np.sign(f(a))+np.sign(f(b))
        if s==0 and np.abs(sign(f(a)))>0 and np.abs(sign(f(b)))>0:
            c = (a+b)/2
            n+=1
            if sign(f(c))==0:
                        a=c
                        b=c
            elif sign(f(c))==sign(f(a)):
                a=c
            elif sign(f(c))==sign(f(b)):
                b=c
        else:
            print("error, wrong start guesses no root in the interval")
            return None
    if err_f(a,b)<tol:
            x=(a+b)/2
            return x
    
phi=(1+np.sqrt(5))/2

def minbracket(f, a, b, n):
    xs = np.linspace(a, b, n+1)
    fs = [f(x) for x in xs]         
    minimas = []
    for i in range(1, n):
        if fs[i] < fs[i-1] and fs[i] < fs[i+1]:
            minimas.append([xs[i-1], xs[i+1]])
    return minimas

def golden_search(f, a, b, tol):
    c = b - (b-a)/phi
    d = a + (b-a)/phi
    fc, fd = f(c), f(d)
    while abs(b-a) > tol:
        if fc < fd:                   
            b, d, fd = d, c, fc       
            c = b - (b-a)/phi
            fc = f(c)                 
        else:                         
            a, c, fc = c, d, fd       
            d = a + (b-a)/phi
            fd = f(d)                 
    return (a+b)/2

def minimum_finder(f, a, b, n, tol):
    brackets = minbracket(f, a, b, n)
    return[golden_search(f, x_min, x_max, tol) for x_min, x_max in brackets]