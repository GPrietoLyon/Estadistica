import numpy as np
from scipy import special as sp
from scipy import stats as st
import matplotlib.pyplot as plt


tiempo=[]
f=[]
M=[]
with open("datos.dat") as fil:
    for i in fil:
        a=i.split(" ")
        tiempo.append(float((a[0])))
        f.append(float((a[1])))


nptiempo=np.array(tiempo)   
npflujo=np.array(f)

"""
Notar que se ignora el transito
"""
i=0
for t in tiempo:    
    l=[1,t,t**2,t**3,t**4,t**5]
    M.append(l)

        
print("calculo")        
npM=np.matrix(M)
npY=np.array(f)
npMT=npM.transpose()
MTM=np.dot(npMT,npM)
MTMinv=np.linalg.inv(MTM)

MTMMT=np.dot(MTMinv,npMT)
th=np.transpose(np.dot(MTMMT,(npflujo)))
theta=[float(th[0]),float(th[1]),float(th[2]),float(th[3]),float(th[4]),float(th[5])]


sigma=3e-5


fit=[]

for t in tiempo:
    fit.append(theta[0]+theta[1]*t+theta[2]*t**2+theta[3]*t**3+theta[4]*t**4+theta[5]*t**5)
    

f1000=[]

"""
Aqui se llevan a cabo las 1000 simulaciones
"""
for i in range(0,1000):
    flujo=[]
    for t in tiempo:
        flujo.append(np.random.normal(0,sigma)+theta[0]+theta[1]*t+theta[2]*t**2+theta[3]*t**3+theta[4]*t**4+theta[5]*t**5)       
    f1000.append(flujo)
    


"""
Se calcula el puntaje de chi cuadrado y el pvalue de cada caso
"""

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




df=294.0


pvalues=[]

for i in chis:
    pvalues.append(1-sp.gammainc(df/2 , i/2) )

#plt.plot(tiempo,fit)
#plt.title("Flujos simulados de modelo sin transito")
#plt.plot(tiempo,f1000[3],".",label="simulacion 3")
#plt.plot(tiempo,f1000[100],".",label="simulacion 100")
#plt.xlabel("tiempo")
#plt.ylabel("Flujo")
#plt.legend()
#plt.show()
#plt.hist(pvalues,bins=100)
#plt.title("p-values de 1000 simulaciones sin transito")
#plt.show()




