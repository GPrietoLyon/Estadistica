import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii
import batman

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





##Covarianza
C=np.cov(np.transpose(datos))
#autocosas
auto=np.linalg.eig(C)
l=auto[0]
V=auto[1]

###Z score

mu=[]
sigma=[]

for i in np.transpose(datos):
    mu.append(np.mean(i))
    sigma.append(np.std(i))
#estrella target
sigmatarget=np.std(target)
meantarget=np.mean(target)
Ztarget=[]
for i in target:
    Ztarget.append((i-meantarget)/sigmatarget)
    
Z=[]

for i in range(0,len(datos)):
    fila=[]
    for j in range(0,len(datos[0])):
        fila.append((datos[i][j]-mu[j])/sigma[j])
    Z.append(fila)


F=np.dot(Z,V)

FT=np.transpose(F)

signal=[]
signal.append(FT[0])
signal.append(FT[1])#Signals importantes
signal.append(FT[2])



def betmen(rp,a,inc):
    par = batman.TransitParams()
    par.t0 = 0.                       #time of inferior conjunction
    par.per = 0.78884                      #orbital period
    par.rp = rp                      #planet radius (in units of stellar radii)
    par.a = a                       #semi-major axis (in units of stellar radii)
    par.inc = inc                    #orbital inclination (in degrees)
    par.ecc = 0.                      #eccentricity
    par.w = 90.                       #longitude of periastron (in degrees)
    par.u = [0.1, 0.3]                #limb darkening coefficients
    par.limb_dark = "quadratic"       #limb darkening model
    t = np.linspace(-2.0/24.0, 2.0/24.0 , 100)
    m = batman.TransitModel(par, t)    #initializes model
    flux = m.light_curve(par)
    return np.log(flux)
    

sigmas=sigma






######
#MCMC
######


t=np.array(t)


def model1(params,t):
    c,rp,a,inc,alfa1,alfa2,alfa3,sigma = params
    return c+betmen(rp,a,inc)+alfa1*signal[0]+alfa2*signal[1]+alfa3*signal[2]+np.random.normal(0,sigma)
    
L=[]
def lnlike1(p, t, y, yerr):
    L.append(-0.5 * np.sum(((y - model1(p, t))/yerr) ** 2))
    return -0.5 * np.sum(((y - model1(p, t))/yerr) ** 2)
    
    
def lnprior1(params):
    c,rp,a,inc,alfa1,alfa2,alfa3,sigma = params
    if (-50 < c < 50 and  0< rp < 20 and -20 < a < 20 and -90 < inc < 90 and -50 < alfa1< 50 and  -50< alfa2 < 50 and -50 < alfa3 < 50 and -50 < sigma < 50):
        return 0.0
    return -np.inf

def lnprob1(p, x, y, yerr):
    lp = lnprior1(p)
    return lp + lnlike1(p, x, y, yerr) if np.isfinite(lp) else -np.inf

#Nuestros datos
y=Ztarget
y=np.array(y)
yerr=sigmatarget
data = (t,y,yerr)


import emcee

initial = np.array([-2.6, 0.1, 1.0, 0.1 , 0.03 , 0.01 ,0.01 ,1])
ndim = len(initial)
nwalkers=100

p0 = [np.array(initial) + 1e-5 * np.random.randn(ndim) for i in xrange(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob1, args=data)


print("Running burn-in...")
p0, _, _ = sampler.run_mcmc(p0, 500)
sampler.reset()


print("Running production...")

param=sampler.run_mcmc(p0, 1000)
target=np.array(target)

samples = sampler.flatchain


####Prueba chi2
"""
chi2=[]
par=[]
for s in samples[np.random.randint(len(samples), size=100)]:
    suma=0
    for i in range(0,len(y)):
        print(i)
        modelo=model1(s,t)
        print modelo[i]
        suma+=((y[i]-modelo[i])**2)/modelo[i]

    chi2.append(suma)
    par.append(s)


for i in range(0,len(chi2)):
    if chi2[i]==max(chi2):
        buscados=par[i]

plt.plot(t,model1(buscados, t))
plt.plot(t,np.log(target))
plt.show()
"""


##########





#Ploteando algunos de los ajustes

i=0
for s in samples[np.random.randint(len(samples), size=15)]:
    plt.figure(i)
    i+=1
    print "Parametros plot:",i,s
    plt.plot(t, model1(s, t), color="#4682b4", alpha=0.3)
    plt.plot(t,np.log(target))
plt.show()



######### Ploteando distribuciones a posteriori
"""

c,rp,a,inc,alfa1,alfa2,alfa3,sigma=[],[],[],[],[],[],[],[]
for parametro in samples:
    c.append(parametro[0])
    rp.append(parametro[1])
    a.append(parametro[2])
    inc.append(parametro[3])
    alfa1.append(parametro[4])
    alfa2.append(parametro[5])
    alfa3.append(parametro[6])
    sigma.append(parametro[7])   

"""




    
"""
plt.figure(1)
plt.title("c")
plt.hist(c,bins=50)
plt.figure(2)
plt.title("rp")
plt.hist(rp,bins=50)
plt.figure(3)
plt.title("a")
plt.hist(a,bins=50)
plt.figure(4)
plt.title("inc")
plt.hist(inc,bins=50)
plt.figure(5)
plt.title("alfa1")
plt.hist(alfa1,bins=50)
plt.figure(6)
plt.title("alfa2")
plt.hist(alfa2,bins=50)
plt.figure(7)
plt.title("alfa3")
plt.hist(alfa3,bins=50)
plt.figure(8)
plt.title("sigma")
plt.hist(sigma,bins=50)

plt.show()
"""
