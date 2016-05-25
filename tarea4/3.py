import numpy as np
from scipy import special as sp
from scipy import stats as st
import matplotlib.pyplot as plt
from pylab import *


AIC=[]
BIC=[]
grados=[0,1,2,3,4,5,6,7,8,9,10,11,12]
pi=np.pi

for grado in grados:
    tiempo=[]
    f=[]
    M=[]
    with open("datos.dat") as fil:
        for i in fil:
            a=i.split(" ")
            tiempo.append(float((a[0])))
            f.append(float((a[1])))
    
    
    npflux=np.array(f)

    M=[]
       
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
    
    #plt.plot(tiempo,fit)
    #plt.plot(tiempo,f,".")    
        
    k=grado
    N=300
    suma=0
    sigma=3*10**(-5)
    for i in range(len(f)):
        suma+=((f[i]-fit[i])**2)/(sigma**2)
    suma=300*log(1.0/(sqrt(2*pi)*30e-6))-0.5*suma
    aic=-2*suma+2*k+((2*k*(k+1))/(N-k-1))
    bic=-2*suma+k*np.log(N)
    AIC.append(aic)
    BIC.append(bic)
    

plt.title("Metodos AIC y BIC")
plt.plot(grados,AIC,label="AIC")
plt.plot(grados,BIC,label="BIC")
plt.legend()
plt.show()   
    