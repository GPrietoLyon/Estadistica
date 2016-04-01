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



'''
PDF
'''

for prob in r:
    mult=combi*((prob)**(18))*((1-prob)**(15))#distribucion binomial
    pXr.append(mult)

    
Sum=np.sum(pXr)#Sera el denominador del teorema de bayes

prX=pXr/Sum #normalizamos

'''
Ploteo PDF
'''
figure(0)
plt.plot(r,prX,'-b')

plt.xlabel(r"$Probabilidad$ $r$")
plt.ylabel(r"$P(r_i|X)$")
plt.title(r"$PDF$  $Teorema$ $de$ $Bayes$")

plt.draw()





'''
CDF
'''
CDF=[] #acumulamos los velores de la PDF para crear una CDF
cumulative=0
for valor in prX:
    cumulative=cumulative+valor
    CDF.append(cumulative)

'''
Ploteo CDF
'''
figure(1)
plt.plot(r,CDF,'-b')


plt.ylim(0,1.01)
plt.xlabel(r"$Probabilidad$ $r$")
plt.ylabel(r"$\sum P(r_i|X)$")
plt.title(r"$CDF$ $Teorema$ $de$ $Bayes$")

plt.draw()

