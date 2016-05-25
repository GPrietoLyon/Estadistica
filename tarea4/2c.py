import numpy as np
from scipy import special as sp
from scipy import stats as st
import matplotlib.pyplot as plt

def datos():
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
        fit.append(theta[0]+theta[1]*t+theta[2]*t**2+theta[3]*t**3+theta[4]*t**4+theta[5]*t**5+1)
        
    
    f1000=[]
    
    for i in range(0,1000):
        flujo=[]
        for t in tiempo:
            flujo.append(np.random.normal(0,sigma)+theta[0]+theta[1]*t+theta[2]*t**2+theta[3]*t**3+theta[4]*t**4+theta[5]*t**5+1)       
        f1000.append(flujo)
    
    return tiempo,f1000



tiempo,f1000=datos()

""" 
Este codigo es igual a anteriores, solo que se adapto para aceptar distintos grados    
"""

grado=5
pvalues=[]
for f in f1000:    
    M=[]
   
    nptiempo=np.array(tiempo)   
    npflux=np.array(f)-1
    
    i=0
    for t in tiempo:
        l=[]
        l.append(1)
        for i in range(0,grado):    
           l.append(t**(i+1))
        M.append(l)
                     
    npM=np.matrix(M)
    npMT=npM.transpose()
    MTM=np.dot(npMT,npM)
    MTMinv=np.linalg.inv(MTM)
    
    MTMMT=np.dot(MTMinv,npMT)
    th=np.transpose(np.dot(MTMMT,(npflux-1)))
    theta=[]
    for i in range(0,grado+1):
        theta.append(float(th[i]))
        
    fit=[]
    for t in tiempo:
        suma=1
        for i in range(0,grado+1):
            suma=suma+theta[i]*t**(i)     
        fit.append(suma)
    
  
    
    
    chi=0
    contador=0
    sigma=9*10**(-10)
    for t in tiempo:
        chi=chi+(npflux[contador]-fit[contador])**2
        contador=contador+1

    chi=chi/sigma
    df=294.0
    pvalue=(1-sp.gammainc(df/2 , chi/2)  )
    pvalues.append(pvalue)
    
    
    
#plt.title("p-values, grado=20")
#plt.hist(pvalues,bins=100)

#plt.plot(tiempo,fit)
#plt.plot(tiempo,npflux,".")
#plt.show()       
    
            
        
    


