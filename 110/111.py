import numpy as np
import matplotlib.pyplot as plt
#Question 1
def f(x,y):
    return(x*(1+4*y**2)) #define the function f(x,y) = dy/dx

def euler(f,y0,x0,h,xmax):
    ycoor=[y0] #starting value of y
    xcoor=[x0] #starting value of x
    while (xcoor[-1]<=xmax-h): # makes sure the last x value is E
        x,y=xcoor[-1], ycoor[-1] # get the last values of x and y
        y = ycoor[-1]+ h*f(x,y) #update y using the Euler method
        xcoor.append(x+h),ycoor.append(y) # append the new values of x and y to the lists
    return(np.array(xcoor),np.array(ycoor)) #return the x and y coordinates as numpy arrays
x,y,h,xmax=0,0,0.1,1 #initial values of x and y, step size h, and end value E
X,Y=euler(f,x,y,h,xmax)
plt.plot(X,Y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Euler Method Approximation')
plt.show()

#Question 2 and 3 since I make the function to start with
def rk2solve(f,y0,x0,h,xmax):
    ycoor=[y0] #starting value of y
    xcoor=[x0] #starting value of x
    while (xcoor[-1]<=xmax-h):
        x,y=xcoor[-1], ycoor[-1] # get the last values of x and y
        k1=f(x,y) #calculate k1 using the function f(x,y)
        k2=f( x + 1/2 *h, y + 1/2 *k1*h) #calculate k2 using the function f(x,y) and the values of x and y
        xcoor.append(x+h),ycoor.append(y+h*k2) # append the new values of x and y to the lists
    return(np.array(xcoor),np.array(ycoor)) #return the x and y coordinates as numpy arrays
    
Xr,Yr=rk2solve(f,x,y,h,xmax) # compare the same function with
xs = np.linspace(0, 1, 200)
plt.plot(Xr,Yr, label='Runge-Kutta Method')
plt.plot(X,Y, label='Euler Method')
plt.plot(xs, 0.5*np.tan(xs**2), 'k--', label='Eksakt')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Runge-Kutta Method Approximation and Euler Method Approximation')
plt.show()


#Question 4
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
        xcoor.append(x+h),ycoor.append(y) # append the new values of x and y to the lists
    return(np.array(xcoor),np.array(ycoor)) #return the x and y coordinates as numpy arrays  
Xr4,Yr4=rk4solve(f,x,y,h,xmax) # compare the same function with
xs = np.linspace(0, 1, 200)
plt.plot(Xr4,Yr4, label='Runge-Kutta 4th Order Method')
plt.plot(Xr,Yr, label='Runge-Kutta 2nd Order Method')
plt.plot(X,Y, label='Euler Method')
plt.plot(xs, 0.5*np.tan(xs**2), 'k--', label='Eksakt')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Runge-Kutta 4th Order Method Approximation, Runge-Kutta 2nd Order Method Approximation and Euler Method Approximation')
plt.show()  
def g (x,y):
    return(x**2+y**2) #define the function g(x,y) = dy/dx
Xg,Yg=euler(g,x,y,h,xmax)
plt.plot(Xg,Yg, label='Euler Method')
Xg_rk2,Yg_rk2=rk2solve(g,x,y,h,xmax)
plt.plot(Xg_rk2,Yg_rk2, label='Runge-Kutta 2nd Order Method')
Xg_rk4,Yg_rk4=rk4solve(g,x,y,h,xmax)
plt.plot(Xg_rk4,Yg_rk4, label='Runge-Kutta 4th Order Method')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Comparison of Euler, Runge-Kutta 2nd Order and Runge-Kutta 4th Order Methods for g(x,y)')
plt.show()


    