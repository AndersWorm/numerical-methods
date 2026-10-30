import numpy as np
from pathlib import Path
import sys
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
def minbracket(f,a,b,n):
    minimas=[]
    dx=(abs(a-b))/n
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

def golden_search(func, a, b, tol):
    phi = (1 + np.sqrt(5))/2
    while abs(b-a) > tol:
        c = b - (b-a)/phi
        d = a + (b-a)/phi
        if func(c) < func(d):
            b = d
        else:
            a = c
    return (a+b)/2

def powell_search(f, x_n, pset, N_max, N, T, tol):
    module_folder = Path(
        "/numerical methods/opgaver/102")
    #path to the folder where the module is located

    sys.path.insert(0, str(module_folder))  #tells python to look in the module folder for imports

    from mysearch import minbracket
    from mysearch import golden_search
    p_n=pset[0] #initial direction vector is the first unit vector in pset
    n_dir=len(pset) #number of direction vectors I have

    def fline(t):
        x=x_n+t*p_n #gives me two values x=x[0] y=x[1]
        z=f(x[0],x[1]) # calculates f(x,y)
        return z

    for c in range(N_max):
        x_0=x_n.copy() #will copy the latest x_n value as my new starting point
        falls=[] #list for all my decrease
        for i in range(n_dir):
            z_i = fline(0) # for calculating the err between f(i) and f(i+1)=z_f
            p_n=pset[i]
            brackets = minbracket(fline, -T,T, N)
            if not brackets: #means if brackets are empty
                    print("no minima found")
                    break
            a,b=brackets[0] #takes the first bracket and assigns it to a and b
            t_min=golden_search(fline,a, b,tol)
            x_n=x_n+t_min*p_n #my new set of coordinates

            z_f=fline(0) #but now it uses x_n calculated above so not the same as z_i
            decrease=abs(z_f-z_i)
            falls.append(decrease)

        u = (x_n-x_0)/np.linalg.norm(x_n-x_0) # direction vector pointing from x_i to x_n
        p_n=u #updates the direction vector to the new direction vector u.
        brackets = minbracket(fline, -T,T, N)
        if not brackets: #means if brackets are empty
            print("no minima found")
            break
        a,b=brackets[0] #takes the first bracket and assigns it to a and b
        t_min=golden_search(fline,a, b,tol)
        x_n=x_n+t_min*p_n #my new set of coordinates i use this one for convergence test

        #convergence
        if np.linalg.norm(x_n-x_0)<tol: #means it converges
            print("minima found at x=",x_n,"f=",fline(0),"after",c+1,"sweeps")
            break
        else:
            i_max=np.argmax(falls) #finds the index with the biggest decrease
            pset[i_max]=u #updates the direction vector in that index with the biggest decrease to the new direction vector u.
    return x_n, fline(0), c+1 #returns the minimum point, the function value at that point and the number of sweeps it took to find it
        
        
def minbracket(f,a,b,n):
    minimas=[]
    dx=(abs(a-b))/n
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

def golden_search(func, a, b, tol):
    phi = (1 + np.sqrt(5))/2
    while abs(b-a) > tol:
        c = b - (b-a)/phi
        d = a + (b-a)/phi
        if func(c) < func(d):
            b = d
        else:
            a = c
    return (a+b)/2

def powell_search(f, x_n, pset, N_max, N, T, tol):
    module_folder = Path(
        "/numerical methods/opgaver/102")
    #path to the folder where the module is located

    sys.path.insert(0, str(module_folder))  #tells python to look in the module folder for imports

    from mysearch import minbracket
    from mysearch import golden_search
    p_n=pset[0] #initial direction vector is the first unit vector in pset
    n_dir=len(pset) #number of direction vectors I have

    def fline(t):
        x=x_n+t*p_n #gives me two values x=x[0] y=x[1]
        z=f(x[0],x[1]) # calculates f(x,y)
        return z

    for c in range(N_max):
        x_0=x_n.copy() #will copy the latest x_n value as my new starting point
        falls=[] #list for all my decrease
        for i in range(n_dir):
            z_i = fline(0) # for calculating the err between f(i) and f(i+1)=z_f
            p_n=pset[i]
            brackets = minbracket(fline, -T,T, N)
            if not brackets: #means if brackets are empty
                    print("no minima found")
                    break
            a,b=brackets[0] #takes the first bracket and assigns it to a and b
            t_min=golden_search(fline,a, b,tol)
            x_n=x_n+t_min*p_n #my new set of coordinates

            z_f=fline(0) #but now it uses x_n calculated above so not the same as z_i
            decrease=abs(z_f-z_i)
            falls.append(decrease)

        u = (x_n-x_0)/np.linalg.norm(x_n-x_0) # direction vector pointing from x_i to x_n
        p_n=u #updates the direction vector to the new direction vector u.
        brackets = minbracket(fline, -T,T, N)
        if not brackets: #means if brackets are empty
            print("no minima found")
            break
        a,b=brackets[0] #takes the first bracket and assigns it to a and b
        t_min=golden_search(fline,a, b,tol)
        x_n=x_n+t_min*p_n #my new set of coordinates i use this one for convergence test

        #convergence
        if np.linalg.norm(x_n-x_0)<tol: #means it converges
            print("minima found at x=",x_n,"f=",fline(0),"after",c+1,"sweeps")
            break
        else:
            i_max=np.argmax(falls) #finds the index with the biggest decrease
            pset[i_max]=u #updates the direction vector in that index with the biggest decrease to the new direction vector u.
    return x_n, fline(0), c+1 #returns the minimum point, the function value at that point and the number of sweeps it took to find it
        
        
