import matplotlib.pyplot as plt
from numpy import *


def ecuacion(o):
    return sin(o)/(1-cos(o))



v=linspace(0,pi/2,1000)


plt.plot(v,ecuacion(v))
plt.show()