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

 
def powell_search(f, x_n, pset, N_max, N, T, tol, scale=None, ftol=None, shrink=True):
    """
    f      : cost function, called as f(x[0], x[1], ...)
    x_n    : starting guess, e.g. [omega0, w]
    pset   : initial directions in DIMENSIONLESS coordinates (usually unit vectors)
    N_max  : max number of sweeps
    N      : number of subintervals in minbracket
    T      : initial search radius (dimensionless, in units of scale)
    tol    : tolerance on step size (dimensionless) and in golden_search
    scale  : typical step size per parameter, e.g. [dOmega, dw]
    ftol   : tolerance on decrease in f (defaults to tol)
    shrink : let the search radius follow the step size
    """
    x_n = np.asarray(x_n, dtype=float)
    scale = np.ones_like(x_n) if scale is None else np.asarray(scale, dtype=float)
    ftol = tol if ftol is None else ftol
    pset = [np.asarray(p, dtype=float).copy() for p in pset]  # do not modify the caller's list
 
    # Work in dimensionless coordinates s = x/scale, so omega0 and w are comparable
    def F(s):
        return f(*(s * scale))
 
    def line_min(s, p, f_s, T_line):
        """1D minimisation of F along s + t*p. Returns (new s, new f, t)."""
        g = lambda t: F(s + t * p)
        brackets = minbracket(g, -T_line, T_line, N)
        if not brackets:
            return s, f_s, 0.0  # no minimum found: stay put instead of breaking
 
        # More than one bracket: choose the one with the lowest f at its midpoint
        if len(brackets) > 1:
            a, b = min(brackets, key=lambda ab: g(0.5 * (ab[0] + ab[1])))
        else:
            a, b = brackets[0]
 
        t_min = golden_search(g, a, b, tol)
        f_new = g(t_min)  # ONE extra RK4 call, reused afterwards
        if f_new >= f_s:  # never accept a step that makes things worse
            return s, f_s, 0.0
        return s + t_min * p, f_new, t_min
 
    s_n = x_n / scale
    f_n = F(s_n)  # f in the current point, kept so it is never recomputed
    T_cur = T
 
    for c in range(N_max):
        s_0, f_0 = s_n.copy(), f_n
        falls, steps = [], []
 
        # Sweep along all directions
        for p in pset:
            s_n, f_new, t = line_min(s_n, p, f_n, T_cur)
            falls.append(f_n - f_new)
            steps.append(abs(t))
            f_n = f_new
 
        d = s_n - s_0
        # Convergence BEFORE normalising d (avoids division by zero)
        if np.linalg.norm(d) < tol or abs(f_0 - f_n) < ftol:
            x_min = s_n * scale
            print("minimum found at x =", x_min, "f =", f_n, "after", c + 1, "sweeps")
            return x_min, f_n, c + 1
 
        # New direction: net displacement of the sweep (dimensionless, so normalising is OK)
        u = d / np.linalg.norm(d)
        s_n, f_n, t = line_min(s_n, u, f_n, T_cur)
        steps.append(abs(t))
 
        # Replace the direction that gave the largest decrease
        pset[int(np.argmax(falls))] = u
 
        # Shrink the search radius as the steps get smaller
        if shrink:
            T_cur = min(T, max(2 * max(steps), 10 * tol))
 
    x_min = s_n * scale
    print("no convergence after", N_max, "sweeps; best x =", x_min, "f =", f_n)
    return x_min, f_n, N_max
def gaussian(t,mu,sigma):
    return( np.exp( -(t-mu)**2 /(2 * sigma**2 )))       
