import numpy as np
import matplotlib.pyplot as plt
def montyn(f,a,b,N):
    rng = np.random.default_rng() # I initialize the random numpy generator, the samples are uniformly distributed
    #which is what i want since I dont want to favorize any points
    I=0
    for i in range(N):
            p=rng.uniform(a,b) #i prefer this since it generates a random number between a and b
            I+=f(p)
    V=abs(b-a)
    return(V/N *I)
f=lambda x: np.sin(x)**2
print(montyn(f,0,np.pi,1000))
#it hits it spot on
I=[]
n=np.linspace(1,1000,1000)
for i in range(1000):
    I.append(montyn(f,0,np.pi,i+1))
plt.plot(n,I)
plt.xlabel("N"); plt.ylabel("I")
plt.title("Monte carlo integration vs N")
plt.show()