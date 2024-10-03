import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def gauss(pars,x):
    amp=pars[0]
    x0=pars[1]
    sig=pars[2]
    return amp*np.exp(-0.5*(x-x0)**2/sig**2)

def numderiv(fun,pars,x,dpar):
    npar=len(pars)
    A=np.zeros([len(x),npar])
    for i in range(npar):
        pnew=pars.copy()
        pnew[i]=pnew[i]+dpar[i]
        fplus=fun(pnew,x)
        pnew[i]=pars[i]-dpar[i]
        fminus=fun(pnew,x)
        A[:,i]=(fplus-fminus)/(2*dpar[i])
    return A

def newton_numderiv(pars,Ninv,fun,data,x,dpar):
    for i in range(10):
        A=numderiv(fun,pars,x,dpar)
        pred=fun(pars,x)
        r=data-pred
        lhs=A.T@Ninv@A
        rhs=A.T@Ninv@r
        shifts=np.linalg.inv(lhs)@rhs        
        chisq=r@Ninv@r
        print('chisq on iteration ',i,' is ',chisq)
        pars=pars+shifts
    return pars,lhs


def  gauss_chisq(pars,x,data,Ninv):
    y=gauss(pars,x)
    r=data-y
    chisq=r@Ninv@r
    return chisq

def run_chain(pars,fun,data,x,dpar,Ninv,L,nsamp=10000):
    chisq=np.zeros(nsamp)
    npar=len(pars)
    chain=np.zeros([nsamp,npar])
    chain[0,:]=pars
    chisq[0]=fun(pars,x,data,Ninv)
    for i in range(1,nsamp):
        pnew=chain[i-1,:]+L@np.random.randn(npar)
        chi_new=fun(pnew,x,data,Ninv)
        prob=np.exp(0.5*(chisq[i-1]-chi_new))
        #accept if a random number is less than this
        if np.random.rand(1)[0]<prob:
            chain[i,:]=pnew
            chisq[i]=chi_new
        else:
            chain[i,:]=chain[i-1,:]
            chisq[i]=chisq[i-1]
    return chain,chisq

x=np.linspace(-5,5,1001)
ptrue=np.asarray([2,1.0,1.5])
y_true=gauss(ptrue,x)
plt.clf()
plt.plot(x,y_true)
plt.show()

noise=0.5
y=y_true+np.random.randn(len(y_true))*noise
plt.plot(x,y,'.')
Ninv=np.eye(len(y))/noise**2

pguess=np.asarray([1.0,0.5,1.5])
y_guess=gauss(pguess,x)
plt.plot(x,y_guess)

dpars=1e-3*np.ones(len(pguess))
pfit,lhs=newton_numderiv(pguess,Ninv,gauss,y,x,dpars)
yfit=gauss(pfit,x)
plt.plot(x,yfit)
#variance of our parameters about the mean is inv(lhs)
par_vars=np.linalg.inv(lhs)  #parameter variances about their mean
L=np.linalg.cholesky(par_vars)
ntrial=50
chi_trial=np.zeros(ntrial)
for i in range(ntrial):
    p_trial=pfit+L@np.random.randn(len(pfit))
    y_trial=gauss(p_trial,x)
    r=y_trial-y
    chi_trial[i]=r@Ninv@r

chain,chivec=run_chain(p_trial,gauss_chisq,y,x,dpars,Ninv,L)
#chain2,chivec2=run_chain(p_trial,gauss_chisq,y,x,dpars,Ninv,10*L)
#chain3,chivec3=run_chain(p_trial,gauss_chisq,y,x,dpars,Ninv,0.1*L)
print('accept probability is ',1-np.mean(np.diff(chivec)==0))
print('vanilla chain mean: ',np.mean(chain,axis=0))
print('vanilla chain std: ',np.std(chain,axis=0))

#let's say we put a prior on the amplitude to be p_true[0] +/- 0.01
p0=ptrue[0]
perr=0.01
wt=np.exp(-0.5*(chain[:,0]-p0)**2/perr**2)
npar=len(ptrue)
for i in range(npar):
    wt_mean=np.sum(chain[:,i]*wt)/np.sum(wt)
    wt_sqr=np.sum(chain[:,i]**2*wt)/np.sum(wt)
    #variance is <x^2>-<x>^2
    psig=np.sqrt(wt_sqr-wt_mean**2)
    print('new value and uncertainty for parameter ',i,' are ',wt_mean,psig)

T=5
chain_hot,chivec_hot=run_chain(p_trial,gauss_chisq,y,x,dpars,Ninv/T**2,L,nsamp=50000)
delta_chi=chivec_hot-chivec_hot.min()
wt=np.exp(-0.5*(T**2-1)*delta_chi)
print('high-T mean: ',np.mean(chain_hot,axis=0))
print('high-T errs: ',np.std(chain_hot,axis=0))
for i in range(npar):
    wt_mean=np.sum(chain_hot[:,i]*wt)/np.sum(wt)
    wt_sqr=np.sum(chain_hot[:,i]**2*wt)/np.sum(wt)
    #variance is <x^2>-<x>^2
    psig=np.sqrt(wt_sqr-wt_mean**2)
    print('new value and uncertainty for parameter ',i,' are ',wt_mean,psig)

wt_sort=wt.copy()
wt.sort()
wt_cumsum=np.cumsum(wt)
wt_cumsum=wt_cumsum/wt_cumsum[-1]

#let's do 5-sigma errors for amplitude
#mywt,myval=np.sortrows(np.hstack([chain_hot[:,0],wt]),0)
