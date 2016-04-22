from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import *


def normalstandard(x):#pdf de la distribucion standard
    return (1.0/sqrt(2*pi))*exp(-x**2/2)
    
valores=np.linspace(-5,5,100) #valores que le daremos a la normal     
    
u=np.random.uniform(-1,1,size=100000) #valores dados entre -1 y 1
inversas=[]

for i in u: 
    inversas.append(sqrt(2)*erfinv(i)) 
    

plt.title(r"Normal v/s $F^{-1}(U)$")
plt.plot(valores,normalstandard(valores),'-g')
plt.hist(inversas,bins=80,normed=True)
plt.show()
