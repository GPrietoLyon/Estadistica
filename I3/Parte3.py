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

auto=np.linalg.eig(C)
l=auto[0]#autovalores
V=auto[1]#autovectores


###Z score

mu=[]
sigma=[]

for i in np.transpose(datos):
    mu.append(np.mean(i))
    sigma.append(np.std(i))

sigmatarget=np.std(target)#estrella target
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
signal.append(FT[1])
signal.append(FT[2])
M=signal



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


import george
from george import kernels
import emcee

def model2(params, t):
    _, _, c,rp,a,inc,alfa1,alfa2,alfa3 = params
    return c+betmen(rp,a,inc)+alfa1*signal[0]+alfa2*signal[1]+alfa3*signal[2]

def lnlike2(p, t, y, yerr):
    a, tau = np.exp(p[:2])
    gp = george.GP(a * kernels.Matern32Kernel(tau))
    gp.compute(t, yerr)
    return gp.lnlikelihood(y - model2(p, t))

def lnprior2(p):
    lna, lntau, c,rp,a,inc,alfa1,alfa2,alfa3 = p
    if (-5 < lna < 5 and  -5 < lntau < 5 and -50 < c < 50 and  0< rp < 20 and -20 < a < 20 and -90 < inc < 90 and -50 < alfa1< 50 and  -50< alfa2 < 50 and -50 < alfa3 < 50):
        return 0.0
    return -np.inf

def lnprob2(p, x, y, yerr):
    lp = lnprior2(p)
    return lp + lnlike2(p, x, y, yerr) if np.isfinite(lp) else -np.inf
    
y=Ztarget
y=np.array(y)
yerr=sigmatarget
data = (t,y,yerr)
nwalkers=100



initial = np.array([0, 0, -2.6, 0.1, 1.0, 0.1 , 0.03 , 0.01 ,0.01 ,1])
ndim = len(initial)
p0 = [np.array(initial) + 1e-8 * np.random.randn(ndim)
      for i in xrange(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2, args=data)

print("Running first burn-in...")
p = p0[np.argmax(lnp)]
sampler.reset()

# Re-sample the walkers near the best walker from the previous burn-in.
p0 = [p + 1e-8 * np.random.randn(ndim) for i in xrange(nwalkers)]
p0, _, _ = sampler.run_mcmc(p0, 250)

print("Running second burn-in...")
p0, _, _ = sampler.run_mcmc(p0, 250)
sampler.reset()

print("Running production...")
sampler.run_mcmc(p0, 1000)









######### Ploteando distribuciones del radio del planeta
"""

rp=[]
for parametro in samples:
    rp.append(parametro[1])

plt.figure(2)
plt.title("rp")

plt.show()
"""





















