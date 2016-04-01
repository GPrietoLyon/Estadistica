import numpy as np
import math as mt
from numpy.random import randint as rand

Lista=[] #Una lista en la que guardaremos la cantidad de casos en que se dieron
         #18 respuestas para la primera opcion
i=0
Iteraciones=1000000
while i<=Iteraciones:
    Random=rand(2,size=33)#Creamos un array de 33 elementos donde por 
    #conveniencia diremos que la primera opcion seran los Random=1
    VecesOpcion1=np.sum(Random)#La suma de los 1 sera cuantos elijen Op 1
            
    if VecesOpcion1==18:#si 18 elijen Op 1
        Lista.append(1)#Si se da agregamos un 1 a Lista
        
    i+=1


Con18=np.array(Lista)
Suma=np.sum(Con18) #veces que se da 18 op1 dentro de las 1000 iteraciones
print('En las 1000 iteraciones vemos',Suma,'casos en que 18 personas elijen quedarse, asumiendo que r=0.5')
Porcentaje=Suma/1000000.0
print("La probabilidad de que 18 personas elijan la primera opcion con r=0.5 es de",Porcentaje)