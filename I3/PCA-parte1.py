import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii

data=ascii.read("dataset_9.dat")
datos=[]
t=[]
target=[]
f1=[]
f2=[]
f3=[]
f4=[]
f5=[]
f6=[]
f7=[]
f8=[]
f9=[]
for elemento in data:
    datos.append(np.log([elemento[2],elemento[3],elemento[4],elemento[5],elemento[6],elemento[7],elemento[8],elemento[9],elemento[10]]))
    t.append(elemento[0])
    target.append(elemento[1])
    f1.append(elemento[2])
    f2.append(elemento[3])
    f3.append(elemento[4])
    f4.append(elemento[5])
    f5.append(elemento[6])
    f6.append(elemento[7])
    f7.append(elemento[8])
    f8.append(elemento[9])
    f9.append(elemento[10])






C=np.cov(np.transpose(datos))

print "MAtriz de covarianza:",C

auto=np.linalg.eig(C)
l=auto[0] #Autovalores
V=auto[1] #Autovectores

print "Autovalores:",l
print "Autovectores:",V


###Z score

mu=[]
sigma=[]

for i in np.transpose(datos):
    mu.append(np.mean(i))
    sigma.append(np.std(i))


Z=[]

### Standarizando
for i in range(0,len(datos)):
    fila=[]
    for j in range(0,len(datos[0])):
        fila.append((datos[i][j]-mu[j])/sigma[j])
    Z.append(fila)
###########
    
######## Signals
F=np.dot(Z,V)
FT=np.transpose(F)
#############







signal=[]
signal.append(FT[0])
signal.append(FT[1])
signal.append(FT[2])
M=signal

plt.title("Signals")
plt.xlabel("tiempo")
plt.figure(1)
plt.plot(t,FT[0])
plt.plot(t,FT[1])
plt.plot(t,FT[2])
plt.plot(t,FT[3])
plt.plot(t,FT[4])
plt.plot(t,FT[5])
plt.plot(t,FT[6])
plt.plot(t,FT[7])
plt.plot(t,FT[8])

plt.figure(2)
plt.title("Small signals")
plt.plot(t,FT[1])
plt.plot(t,FT[2])
plt.plot(t,FT[3])
plt.plot(t,FT[4])
plt.plot(t,FT[5])
plt.plot(t,FT[6])
plt.plot(t,FT[7])
plt.plot(t,FT[8])


##Important signals

signal=[]
signal.append(FT[0])
signal.append(FT[1])
signal.append(FT[2])
M=signal

###########Estrella a modelar
tar=f1
############################

MT=np.transpose(M)
MTM=np.dot(MT,M)
MTMinv=np.linalg.inv(MTM)
MTMinvMT=np.dot(MTMinv,MT)
theta=np.dot(np.transpose(MTMinvMT),tar)

aparam=theta## Parametros parte: a1,a2,a3
print "a1,a2,a3:",aparam

plt.show()



