from mpl_toolkits.mplot3d import *
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from matplotlib import cm

ptos=[]

with open("datos_clasificacion.dat") as f:
    for i in f:
        a=i.split(" ")
        b=a[0].split("\t")
        ptos.append([float((b[0])),float((b[1])),float((a[1]))])




for dato in ptos:
    if dato[2]==1:
        plt.plot(dato[0],dato[1],".",color='red')
    elif dato[2]==2:
        plt.plot(dato[0],dato[1],".",color='blue')
        
        
regresX=np.linspace(-5,12,100)
regresY=-1.0850*regresX+8.5572

plt.title("Regresion lineal")

plt.plot(regresX,regresY)        
        
        
        
        
        
plt.show()
        


"""
xx=0
xy=0
x=0
yy=0
y=0
N=0
xz=0
yz=0
z=0

for dato in ptos:
    xx+=dato[0]**2
    yy+=dato[1]**2
    xy+=dato[0]*dato[1]
    x+=dato[0]
    y+=dato[1]
    N+=1
    xz+=dato[0]*dato[2]
    yz+=dato[1]*dato[2]
    z+=dato[2]

a=[[xx,xy,x],[xy,yy,y],[x,y,N]]
b=[xz,yz,z]

sol=np.linalg.solve(a,b)

fig = plt.figure()
ax = fig.gca(projection='3d')              
plt.hold(True)

x_surf=np.arange(-10, 10, 1)                
y_surf=np.arange(-10, 10, 1)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_surf =sol[0]*x_surf+sol[1]*y_surf+sol[2]         
ax.plot_surface(x_surf, y_surf, z_surf);    
x=[]
y=[]
z=[]
  
for i in ptos:
    x.append(i[0])
    y.append(i[1])
    z.append(i[2])


ax.scatter(x, y, z);                      

ax.set_xlabel('x label')
ax.set_ylabel('y label')
ax.set_zlabel('z label')

plt.show()

"""