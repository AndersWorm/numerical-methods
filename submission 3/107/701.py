import numpy as np
import numpy as np
from pathlib import Path
import sys

module_folder = Path(__file__).resolve().parent.parent / "myfunctions" #path to the folder where the module is located

sys.path.insert(0, str(module_folder))  #tells python to look in the module folder for imports

from myquad import quadz

def squad2d(f,a,b,c,d,tol=1e-5):
    def yint(x): #I define my y integral which quadz will use for different values of x in the end
        def fy(y): #I define my dummy function fy which is f(x,y) but only y varies
            return(f(x,y))
        Iy,n=quadz(fy,c,d,tol) #,n becausse my quadz also returns n
        return(Iy)
    I,n=quadz(yint,a,b,tol)
    return(I)
#the code calculates yint for different values of x given by quadz and then quadz integrate that over x

f = lambda x,y: x**2 * y
a,b,c,d=0,2,0,1
result=squad2d(f,a,b,c,d)
print(result)

#if I follow the text I get that result howeveer the integral suggest that I should do it the other way arroung
#and first integrate x and thereafter do y
result=squad2d(f,c,d,a,b)  
print(result)
#if I do it as the integral I get 2/3 instead of 4/3

#test new function 
f = lambda x,y: np.cos(x**2) +  np.sin(y)
a,b,c,d=0,2*np.pi,0,np.pi/2
result=squad2d(f,a,b,c,d)
print(result)

p=np.random.uniform(size=10)