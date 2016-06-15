import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

clase1=[]
clase2=[]
ptos=[]
plot=[]
with open("datos_clasificacion.dat") as f:
    for i in f:
        a=i.split(" ")
        b=a[0].split("\t")
        ptos.append([float(b[0]),float(b[1])])
        plot.append([float(b[0]),float(b[1]),float(a[1])])        
        if float(a[1])==1:
            clase1.append([float((b[0])),float((b[1]))])
        if float(a[1])==2:
            clase2.append([float((b[0])),float((b[1]))])
            
ptos=np.transpose(ptos)
clase1=np.transpose(clase1)
clase2=np.transpose(clase2)
C1=np.cov(clase1)
C2=np.cov(clase2)
u1=[]
u2=[]
u1.append(sum(clase1[0])/300.0)
u1.append(sum(clase1[1])/300.0)
u2.append(sum(clase2[0])/500.0)
u2.append(sum(clase2[1])/500.0)
u1=np.matrix(u1)
u2=np.matrix(u2)

n1=300.0
n2=500.0
p1=n1/(n1+n2)
p2=n2/(n1+n2)

C1inv=np.linalg.inv(C1)
C2inv=np.linalg.inv(C2)

def func(x,u,Cinv,p):
    xu=-0.5*(x-u)
    xuC=np.dot(xu,Cinv)
    xuCxu=np.dot(xuC,np.transpose(x-u))
    lnC=-0.5*np.log(np.linalg.det(np.linalg.inv(Cinv)))
    z=xuCxu+lnC
    return z
    
    
#Aqui vemos que tantos tipo 1 quedan clasificados como tipo 1    
    
"""    
x=[]
y=[]
Z1=[]
Z2=[]
for coord in plot:
    x.append(coord[0])
    y.append(coord[1])
    
for i in range(0,len(x)):
    xx=[x[i],y[i]]
    Z1.append(func(xx,u1,C1inv,p1))
    Z2.append(func(xx,u2,C2inv,p2))

contador=0
for i in range(0,len(Z1)):
    if Z2[i]>Z1[i]:
        contador+=1
        print contador
        
"""    



##Aqui se calcula la curva de decision
"""
En pocas palabras, lo que se hizo para trazar fue encontrar para cada
columna de una grilla que representa el plano el punto donde las probabilidades
de pertenecer al grupo 1 o 2 se hacian practicamente iguales y por lo tanto
su diferencia se volvia cero.    
De este modo se puede formar una curva de separacion
"""    

f1=[]
f2=[]
for i in plot:    
    x=[i[0],i[1]]
    f1.append(func(x,u1,C1inv,p1))
    f2.append(func(x,u2,C2inv,p2))   
    
x=np.linspace(-2,8,200)
y=np.linspace(-2,8,200)

pto=[]
cercano=[]
for i in range(0,len(x)):
    a=[]
    coord=[]
    f=[]
    for j in range(0,len(x)):
        xx=[x[i],y[j]]
        f.append([func(xx,u1,C1inv,p1),func(xx,u2,C2inv,p2),xx])
    
    for elemento in f:
        a.append(np.abs(float(elemento[0])-float(elemento[1])))
        coord.append(elemento[2])
        
    m=min(a)
    for i in range(0,len(a)):
        if a[i]==m:
            cercano.append([m,coord[i]])
    
print(cercano)
        

print(cercano[0])
losx=[]
losy=[]
for coords in cercano:
    cor=coords[1]
    losx.append(cor[0])
    losy.append(cor[1])



plt.plot(losx,losy)
for dato in plot:
    if dato[2]==1:
        plt.plot(dato[0],dato[1],".",color='red')
    elif dato[2]==2:
        plt.plot(dato[0],dato[1],".",color='blue')
  
plt.title("QDA")
plt.show()

    
    
    
    
    