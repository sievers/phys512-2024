import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def get_step(trial_step):
    if len(trial_step.shape)==1:
        return np.random.randn(len(trial_step))*trial_step
    else:
        L=np.linalg.cholesky(trial_step)
        return L@np.random.randn(trial_step.shape[0])


def gauss(pars,x):
    amp=pars[0]
    x0=pars[1]
    sig=pars[2]
    y=amp*np.exp(-0.5*(x-x0)**2/sig**2)
    return y

def gauss_chisq(pars,data):
    x=data['x']
    y=data['y']
    errs=data['errs']

    pred=gauss(pars,x)
    chisq=np.sum((pred-y)**2/errs**2)
    return chisq

def num_derivs(fun,pars,dp,x):
    A=np.empty([len(x),len(pars)])
    for i in range(len(pars)):
        pp=pars.copy()
        pp[i]=pars[i]+dp[i]
        y_right=fun(pp,x)
        pp[i]=pars[i]-dp[i]
        y_left=fun(pp,x)
        A[:,i]=(y_right-y_left)/(2*dp[i])
    return A
def newton(fun,pars,dp,x,y,niter=10):
    for i in range(niter):
        pred=fun(pars,x)
        r=y-pred
        A=num_derivs(fun,pars,dp,x)
        lhs=A.T@A
        rhs=A.T@r
        step=np.linalg.inv(lhs)@rhs
        pars=pars+step
        print('step is ',step)
    return pars,np.linalg.inv(lhs)

def run_chain(fun,pars,trial_step,data,nstep=20000,T=1):
    npar=len(pars)
    chain=np.zeros([nstep,npar])
    chisq=np.zeros(nstep)
    chain[0,:]=pars
    chi_cur=fun(pars,data)
    chisq[0]=chi_cur
    for i in range(1,nstep):
        pp=pars+get_step(trial_step)
        new_chisq=fun(pp,data)
        accept_prob=np.exp(-0.5*(new_chisq-chi_cur)/T)
        if np.random.rand(1)<accept_prob:
            pars=pp
            chi_cur=new_chisq
        chain[i,:]=pars
        chisq[i]=chi_cur
    return chain,chisq

def process_chain(chain,chisq,T=1.0):
    dchi=chisq-np.min(chisq)
    #density in chain is exp(-0.5*chi^2/T), but
    #we wanted it to be exp(-0.5*chi^2)
    #so, we want to downweight by ratio, which is
    #exp(-0.5*chi^2*(1-1/T)).  We'll calculate the mean
    #and standard deviation of the chain, but will also
    #return the weights so you could calculate whatever you want

    wt=np.exp(-0.5*dchi*(1-1/T)) #the magic line that importance samples

    #calculate the weighted sum of the chain and the chain squared
    npar=chain.shape[1]
    tot=np.zeros(npar)
    totsqr=np.zeros(npar)
    for i in range(npar):
        tot[i]=np.sum(wt*chain[:,i])
        totsqr[i]=np.sum(wt*chain[:,i]**2)
    #divide by sum or weights
    mean=tot/np.sum(wt)
    meansqr=totsqr/np.sum(wt)

    #variance is <x^2>-<x>^2
    var=meansqr-mean**2
    return mean,np.sqrt(var),wt
    


x=np.linspace(-5,5,1001)
amp=1.0
x0=0.0
sig=1.0
y_true=amp*np.exp(-0.5*(x-x0)**2/sig**2)
noise=0.01
N=noise**2
y=y_true+np.random.randn(len(x))*noise

pars=np.asarray([amp,x0,sig])
dp=np.asarray([0.01,0.0001,0.00001])
fitp,curve=newton(gauss,pars,dp,x,y)
curve=curve*N #since we didn't include the noise in Newton's method
data={}
data['x']=x
data['y']=y
data['errs']=noise

chain,chivec=run_chain(gauss_chisq,fitp,curve,data)
nsig=5
T=nsig**2
chain2,chivec2=run_chain(gauss_chisq,fitp,curve*T,data,T=T)

mean,errs,wts=process_chain(chain,chivec)
mean2,errs2,wts2=process_chain(chain2,chivec2,T)

npar=chain.shape[1]
for i in range(npar):
    t1=mean[i]+errs[i]*nsig
    t2=mean[i]-errs[i]*nsig
    frac=(np.sum(chain[:,i]>t1)+np.sum(chain[:,i]<t2))/chain.shape[0]
    frac2=(np.sum(chain2[:,i]>t1)+np.sum(chain2[:,i]<t2))/chain2.shape[0]
    print('fractions of samples on param ',i,' more than ',nsig,' is ',frac,frac2)
    
