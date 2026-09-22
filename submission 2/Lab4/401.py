import numpy as np
import matplotlib.pyplot as plt
#creates linspace
x=np.linspace(0,2,5)
y=np.linspace(0,5,10)
X,Y=np.meshgrid(x,y) 
#makes 2 5x10 arrays where X is constant
#over the colloumns and Y is constant over the rows
#for it to becomes 100x100 I need 100 points in each direction.
x=np.linspace(0,2,100)
y=np.linspace(0,5,100)
X,Y=np.meshgrid(x,y)
plt.pcolor(X,Y,np.exp(-X**2-Y**2))
plt.show()
#plots e^-(x^2+y^2) using the points from linspace