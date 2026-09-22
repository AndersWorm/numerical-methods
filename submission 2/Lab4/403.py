import numpy as np
import matplotlib.pyplot as plt

def f(x,y):
    return (x-1)**2 + y**2
x=np.linspace(-2,2,100)
y=np.linspace(-1,1,100)
#taking 100 points in each interval
X,Y=np.meshgrid(x,y)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, f(X,Y))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
plt.show()


x_n=X[0,0],Y[0,0] # the first point in X is (-2) and first in Y is -1
h=0.2
p_n=np.array([1,0]) # takes 1 step in x zero in y.
X_f=[] #creates a list for the updated x,y values
for i in range(20):
    x_n=x_n+ h * p_n
    X_f.append(x_n)
X_f=np.array(X_f) #array with the updated points in XY [:,0] means 1. colloumn
Z_f=f(X_f[:,0],X_f[:,1]) # the Z values in the points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') # creates a 3D frame
surf = ax.plot_surface(X, Y, f(X,Y)) #plots the surface on the 3D frame
dots=ax.scatter(X_f[:,0],X_f[:,1],Z_f,color='red', s=40, depthshade=False) # adds the points on the surface
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
plt.show()

x_n = np.array([X[0,0], Y[0,0]])   #make sure it starts with this one [-2,-1]
def fline(t):
    x=x_n+t*p_n #gives me two values x=x[0] y=x[1]
    z=f(x[0],x[1]) # calculates f(x,y)
    return z

from pathlib import Path
import sys

module_folder = Path(
    "/Users/andersworm/Downloads/numerical methods/opgaver/102"
) #path to the folder where the module is located

sys.path.insert(0, str(module_folder))  #tells python to look in the module folder for imports

from mysearch import minbracket
from mysearch import golden_search
brackets = minbracket(fline, 0,h*20, 20) # minbrackets uses fline and run through t=0 to t=20h in 20 steps
a,b=brackets[0] #takes the first bracket and assigns it to a and b
tol=1e-5
minima=fline(golden_search(fline,a, b,tol))

print("The minimum value of f along the line is:", minima,"±",tol)
