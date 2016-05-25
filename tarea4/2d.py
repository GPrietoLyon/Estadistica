import numpy as np
from scipy import special as sp
from scipy import stats as st
import matplotlib.pyplot as plt

"""
Tomamos directamente los coeficientes obtenidos en la pregunta 1
"""


theta=[-0.00113006,0.01220323,-0.04505635,0.07422477,-0.05635102,0.01623969]
d=-0.000101

tiempo=[]
with open("datos.dat") as f:
    for i in f:
        a=i.split(" ")
        tiempo.append(float((a[0])))


sigma=3.0e-5
"""
nuevamente se considera transito
"""
fit=[]

for t in tiempo:
    if t<0.4 or t>0.7:
        fit.append(theta[0]+theta[1]*t+theta[2]*t**2+theta[3]*t**3+theta[4]*t**4+theta[5]*t**5+1)
    elif t>=0.4 and t<=0.7:
        fit.append(theta[0]+theta[1]*t+theta[2]*t**2+theta[3]*t**3+theta[4]*t**4+theta[5]*t**5+1+d)


f1000=[]

for i in range(0,1000):
    flujo=[]
    for t in tiempo:
        if t<0.4 or t>0.7:
            flujo.append(np.random.normal(0,sigma)+theta[0]+theta[1]*t+theta[2]*t**2+theta[3]*t**3+theta[4]*t**4+theta[5]*t**5+1)       
        elif t>=0.4 or t<=0.7:
            flujo.append(np.random.normal(0,sigma)+theta[0]+theta[1]*t+theta[2]*t**2+theta[3]*t**3+theta[4]*t**4+theta[5]*t**5+1+d)       
            
    f1000.append(flujo)
    print(i)







chis=[]
chi=0
contador=0

sigma=9.0*10**(-10)

for i in range(0,1000):  
    
    for t in tiempo:
        flujo=f1000[i]
        chi=chi+((flujo[contador]-fit[contador])**2)/sigma
        contador=contador+1

    chis.append(chi)
    chi=0
    contador=0




df=293.0



pvalues=[]

for i in chis:
    pvalues.append(1-sp.gammainc(df/2.0 , i/2.0) )


#plt.plot(tiempo,fit)
#plt.plot(tiempo,f1000[0],".")
#plt.show()




plt.hist(pvalues,bins=100)
plt.title("p-values modelo con transito")
plt.show()






