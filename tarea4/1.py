import numpy as np
from scipy import special as sp
from scipy import stats as st
import matplotlib.pyplot as plt

tiempo=[]
flujo=[]
M=[]

"""
Leyendo el documento

"""
with open("datos.dat") as f:
    for i in f:
        a=i.split(" ")
        tiempo.append(float((a[0])))
        flujo.append(float((a[1])))
        
        
nptiempo=np.array(tiempo)   
npflujo=np.array(flujo)
i=0

"""
Creando la matriz M y calculando MLE

"""

for t in tiempo:
    if t>=0.4 and t<=0.7:
        l=[1,t,t**2,t**3,t**4,t**5,1]
        M.append(l)
    elif t<0.4 or t>0.7:
        l=[1,t,t**2,t**3,t**4,t**5,0]
        M.append(l)
npM=np.matrix(M)
npY=np.array(flujo)
npMT=npM.transpose()
MTM=np.dot(npMT,npM)
MTMinv=np.linalg.inv(MTM)

MTMMT=np.dot(MTMinv,npMT)
theta=np.transpose(np.dot(MTMMT,(npflujo-1)))

fit=[]

print(theta[6])

"""
Usando los coeficientes y la condicion de transito (delta)

"""


for x in tiempo:
    if x<0.4 or x>0.7:
        fit.append(1+float(theta[0])+float(theta[1])*x+float(theta[2])*x**2+float(theta[3])*x**3+float(theta[4])*x**4+float(theta[5])*x**5)
    elif x>=0.4 and x<=0.7:
        fit.append(1+float(theta[6])+float(theta[0])+float(theta[1])*x+float(theta[2])*x**2+float(theta[3])*x**3+float(theta[4])*x**4+float(theta[5])*x**5)
        

"""
Calculando chi cuadrado

"""


chi=0
contador=0
sigma=9*10**(-10)
for t in tiempo:
    chi=chi+(flujo[contador]-fit[contador])**2
    contador=contador+1
    
    
"""
Calculando los pvalues a partir de la cdf

"""    
  
chi=chi/sigma
df=293.0
x=np.linspace(0,1000,1000)

chi2=sp.gammainc(df/2 , x/2)   
print(theta)
def chi2(x):
    chi2=sp.gammainc(df/2 , x/2)   
    return chi2

print(chi)
pvalue=(1-chi2(chi))
print(pvalue)
#plt.plot(x,chi2(x))
#plt.show()

plt.plot(tiempo,fit)
plt.plot(tiempo,flujo,".")
plt.title("Fiteando un polinomio grado 5")
plt.xlabel("Tiempo")
plt.ylabel("Flujo")
plt.show()
    

    
    












