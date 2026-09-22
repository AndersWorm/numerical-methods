import numpy as np
import matplotlib.pyplot as plt
def polyx(a,x): #creates a polynomial function which sums up all the coefficients.
    f=0 
    for i in range(len(a)): # I use the index to pair the coeficients with x, by using len(a) it can be used for n type polynomials.
        f+=a[i]* x**(i)
    return f

xp=np.linspace(-1,2,4).reshape(-1,1) # make an array with 1 coloumn an as many rows as needed
yp=np.array([[2],[-1],[1],[0]])

A=np.hstack([xp,xp]) #hstack combines the two matrices from two 4x1 matrices to one 4x2 matrix

n=len(xp)
V=np.zeros((n,n)) #nXn dimensioned matrix with zeroes I can update
for j in range(n): # I sum over the rows with j constant as V says, find it much easier than Hstack.
    for i in range(n):
        V[i,j]=xp[i,0]**(j) #xp[i,0] means first index in each row i
VI=np.linalg.inv(V)
C=np.matmul(VI,yp)
x=np.linspace(-1,2,100)
Y=polyx(C,x)


x=np.linspace(-1,2,100) #linearly spaced values between -1 and 2, 100 values
Y=polyx(C,x) #find the polynomial function with the four coeficients and the 100 values of x.
plt.plot(x,Y) #plot the polynomial function
plt.scatter(xp,yp,color='r') #plot the four points
plt.title('Polynomial function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['Polynomial function', 'Points'])
plt.grid()
plt.show()

