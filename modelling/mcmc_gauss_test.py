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


def mygauss(pars,x):
    amp=pars[0]
    x0=pars[1]
    sig=pars[2]
    return amp*np.exp(-0.5*((x-x0)**2/sig**2))

def mygauss_chisq(pars,x,y,noise=1):
    mymod=mygauss(pars,x)
    return np.sum( (y-mymod)**2/noise**2)

def num_derivs(fun,pars,x,dpars=1e-3):
    if isinstance(dpars,type(1.0)):
        dpars=0*pars+dpars
    npar=len(pars)
    grad=np.zeros([len(x),npar])
    for i in range(npar):
        pp=pars.copy()
        pp[i]=pp[i]-dpars[i]
        f_left=fun(pp,x)
        pp[i]=pars[i]+dpars[i]
        f_right=fun(pp,x)
        grad[:,i]=(f_right-f_left)/(2*dpars[i])
    return grad
def newton(pars,fun,x,d,Ninv,dpars,chitol=0.01):
    maxiter=10
    oldchi=1e99
    for i in range(maxiter):
        Am=fun(pars,x)
        r=d-Am
        chisq=r.T@Ninv@r
        print('on iteration ',i,' chisq is ',chisq)
        grad=num_derivs(fun,pars,x,dpars)
        lhs=grad.T@Ninv@grad
        rhs=grad.T@Ninv@r
        pshift=np.linalg.inv(lhs)@rhs
        pars=pars+pshift
        dchi=oldchi-chisq
        oldchi=chisq
        if np.abs(dchi)<chitol:
            print('Newton converged after ',i,' iterations.')
            return pars,lhs
    print('failed to converge, last dchi was ',np.abs(dchi))
    return pars
                                    

def run_mcmc(pars,x,data,noise,nsamp,chi_fun,dstep,L=None):
    npar=len(pars)
    chain=np.zeros([nsamp,npar])
    chisq=chi_fun(pars,x,data,noise)
    chivec=np.zeros(nsamp)
    for i in range(nsamp):
        if L is None:
            pnew=pars+np.random.randn(npar)*dstep
        else:
            pnew=pars+L@np.random.randn(npar)
        chi_new=chi_fun(pnew,x,data,noise)
        like_ratio=np.exp(-0.5*(chi_new-chisq))
        if like_ratio>np.random.rand(1)[0]:
            pars=pnew
            chisq=chi_new
        chain[i,:]=pars
        chivec[i]=chisq
    return chain,chivec
n=1000
x=np.linspace(-5,5,n+1)
amp=2
x0=1
sig=1.5
pars_true=np.asarray([amp,x0,sig])
pguess=np.asarray([0.8*amp,0.5,1.2])
y_true=mygauss(pars_true,x)

noise=0.3
y=y_true+np.random.randn(len(y_true))*noise
Ninv=np.eye(len(y))/noise**2
dpar=1e-3
fitp,curve=newton(pguess,mygauss,x,y,Ninv,dpar)
pred=mygauss(fitp,x)
plt.clf()
plt.plot(x,y,'.')
plt.plot(x,pred)
plt.show()

L=np.linalg.cholesky(np.linalg.inv(curve))

#noise=1/np.sqrt(np.diag(Ninv))
samps,chivec=run_mcmc(pguess,x,y,noise,50000,mygauss_chisq,0,L.T*1.5)
print('mcmc params: ',np.mean(samps,axis=0))

T=15
samps_high,chivec_high=run_mcmc(pguess,x,y,noise*T,150000,mygauss_chisq,0,L.T*1.5*T)
ncut=1000
samps_high=samps_high[ncut:,:]
chivec_high=chivec_high[ncut:]
chivec_shift=chivec_high-chivec_high.min()
chi_true=chivec_shift*T**2
chi_wt=chi_true-chivec_shift
wt=np.exp(-0.5*chi_wt)

mean_weighted=np.sum(samps_high*np.outer(wt,np.ones(len(fitp))),axis=0)/np.sum(wt)
var_weighted=np.sum(samps_high**2*np.outer(wt,np.ones(len(fitp))),axis=0)/np.sum(wt)
errs_weighted=np.sqrt(var_weighted-mean_weighted**2)
print('low T errs: ',np.std(samps[ncut:,:],axis=0))
print('high T wt errs: ',errs_weighted)
print('high T ',T,'-sigma: ',np.std(samps_high,axis=0))
print('high T extrap: ',np.std(samps_high,axis=0)/T)

