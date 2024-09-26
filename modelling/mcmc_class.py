import numpy as np
from matplotlib import pyplot as plt

def mysin(pars,x):
    amp=pars[0]
    phase=pars[1]
    nu=pars[2]
    return amp*np.sin(nu*x/2/np.pi+phase)

def mysin_chisq(pars,data,noise=1):
    sin=mysin(pars,np.arange(len(data)))
    r=data-sin
    chisq=np.sum((r/noise)**2)
    return chisq


def run_mcmc(pars,data,nsamp,chi_fun,dstep):
    npar=len(pars)
    chain=np.zeros([nsamp,npar])
    chisq=chi_fun(pars,data)
    chivec=np.zeros(nsamp)
    for i in range(nsamp):
        pnew=pars+np.random.randn(npar)*dstep
        chi_new=chi_fun(pnew,data)
        like_ratio=np.exp(-0.5*(chi_new-chisq))
        if like_ratio>np.random.rand(1)[0]:
            pars=pnew
            chisq=chi_new
        chain[i,:]=pars
        chivec[i]=chisq
    return chain,chivec
n=1000
x=np.arange(n)
nu=0.1
amp=0.2
phase=0
pars=np.asarray([amp,phase,nu])
y_true=mysin(pars,x)
plt.clf()
plt.plot(y_true)
plt.show()
y=y_true+np.random.randn(n)
plt.plot(y,'.')


samps,chivec=run_mcmc(pars,y,50000,mysin_chisq,np.asarray([0.01,0.01,0.001]))

