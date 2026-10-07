import numpy as np
import matplotlib.pyplot as plt

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

#Question 5 difference is y f and k becomes vectors but the code is the same as before i just need to change the function f to take in a vector y and return a vector dy/dx
y0 = [1,2]
x0=0
def f(x,y):
    y1,y2=y # takes only 1 argument for y so I dont have to change rk4 but y does now contain more coordinates
    dy1=y1**2+x*y2-1
    dy2=y1*y2-x*y1
    return(np.array([dy1,dy2]))
x,y,h,xmax=0,0,0.1,1 #initial values of x and y, step size h, and end value E      
X, Y = rk4solve(f, y0, x0, h, xmax)
plt.plot(X, Y[:, 0], label='y1')#since y is now a vector I can plot the first coordinate of y with Y[:,0] and the second with Y[:,1]
plt.plot(X, Y[:, 1], label='y2') # y2 is the second coordinate of y with Y[:,1] the derivative of y2
plt.show()
#So the code will still work for Q1-4
# Question 6
# y2+4y=0 makes to a system of 2 first order ODEs by letting y1=y and y2=y' so that y1'=y2 and y2'=-4y1
def g(x,y):
    y1,y2=y # takes only 1 argument for y so I dont have to change rk4 but y does now contain more coordinates
    dy1=y2 #equation for y1' = y2
    dy2=-4*y1 #equation for y2' = -4y1
    return(np.array([dy1,dy2]))
X, Y = rk4solve(g, [1,0], 0, 0.1, 10) #initial values of y1 and y2 are 1 and 0 respectively
xs = np.linspace(0, 10, 200)
plt.plot(X, Y[:, 0], label='y1')#since y is now a vector I can plot the first coordinate of y with Y[:,0] and the second with Y[:,1]
plt.plot(X, Y[:, 1], label='y2')
plt.plot(xs, np.cos(2*xs), 'k--', label='Eksakt y1') #exact solution for y1 is cos(2x)
plt.plot(xs, -2*np.sin(2*xs), 'k:', label='Eksakt y2') #exact solution for y2 is -2sin(2x) the derivative of cos(2x) is -2sin(2x)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Runge-Kutta Method Approximation for System of ODEs')
plt.legend()
plt.show()