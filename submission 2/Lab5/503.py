import numpy as np
import matplotlib.pyplot as plt
def polyx(a,x): #creates a polynomial function which sums up all the coefficients.
            f=0 
            for i in range(len(a)): # I use the index to pair the coeficients with x, by using len(a) it can be used for n type polynomials.
                f+=a[i]* x**(i)
            return f
def vandint(x,xp,yp):    
        n=len(xp)
        V=np.zeros((n,n)) #nXn dimensioned matrix with zeroes I can update
        for j in range(n): # I sum over the rows with j constant as V says, find it much easier than Hstack.
            for i in range(n):
                V[i,j]=xp[i,0]**(j) #xp[i,0] means first index in each row i
        VI=np.linalg.inv(V)
        C=np.matmul(VI,yp)
        return(polyx(C,x))
xp=np.linspace(-1,2,4).reshape(-1,1) # make an array with 1 coloumn an as many rows as needed
yp=np.array([[2],[-1],[1],[0]])
x=np.linspace(-1,2,100)
Y=vandint(x,xp,yp)
plt.plot(x,Y)
plt.scatter(xp,yp,color='r')
plt.title('Polynomial function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['Polynomial function', 'Points'])
plt.show()