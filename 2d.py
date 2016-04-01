import numpy as np
import math as mt
import matplotlib.pyplot as plt
from numpy.random import randint as rand
from scipy.special import comb
from matplotlib import rc
import matplotlib as mpl
from pylab import *

r=np.linspace(0,1,1000)#lista que llenaremos de distintos valores de r
prX=[]#lista que usaremos de eje Y p(ri|X)
pXr=[]#lista que contendra los p(X|ri)
combi=comb(33,18)#combinatoria de la distribucion binomial
menor=[]#lista con P(r<0.5|r)
mayor=[]#lista con P(r>0.5|r)

'''
PDF
'''

for prob in r:
    mult=combi*((prob)**(18))*((1-prob)**(15))#distribucion binomial
    pXr.append(mult)
    if prob<0.5:
        menor.append(mult)
    if prob>0.5:
        mayor.append(mult)

    
Sum=np.sum(pXr)#Sera el denominador del teorema de bayes




print("La probabilidad de P(r<0.5|r) es", np.sum(menor/Sum))
print("La probabilidad de P(r>0.5|r) es", np.sum(mayor/Sum))
print("La suma de ambas da",np.sum(menor/Sum)+np.sum(mayor/Sum))


