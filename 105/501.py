import numpy as np
import matplotlib.pyplot as plt
def polyx(a,x): #creates a polynomial function which sums up all the coefficients.
    f=0 
    for i in range(len(a)): # I use the index to pair the coeficients with x, by using len(a) it can be used for n type polynomials.
        f+=a[i]* x**(i)
    return f
x=np.linspace(-3,3,50) # 50 values between -3 to 3
a=np.array([1,-2,1])
f_r=1-2*x+x**2 #the real function

plt.scatter(x,polyx(a,x),color='r') #my function calculated with polyx
#colour blue on poly
plt.plot(x, f_r) # the real function
plt.title('Polynomial function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['Polynomial function', 'f(x) = 1-2x+x^2'])
plt.grid()
plt.show()