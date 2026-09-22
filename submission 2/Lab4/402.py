import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(-2,2,100)
y=np.linspace(-1,1,100)
#taking 100 points in each interval
X,Y=np.meshgrid(x,y)
plt.pcolor(X,Y,(X-1)**2 + Y**2) #use pcolor to plot the function (X-1)^2 + Y^2
plt.title('pcolor plot of (X-1)^2 + Y^2')
plt.colorbar() #add a colorbar to the plot
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()