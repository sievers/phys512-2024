import numpy as np

def manygauss(pars,n):
    x=np.arange(n)
    per=pars[0]
    sig=pars[1]
    phase=pars[2]
    amp=pars[3]
    alpha=sig #you should put in your analytic estimate for alpha here
    if alpha==sig:
        print('you need to put in your answer from part A here.')
        assert(1==0)
    y=np.exp(alpha**2*(np.cos( (x-phase)*2*np.pi/per)-1))
    return 1-amp*y


def numgrad(fun,pars,dpar,n):
    #calculate numerical derivatives of function fun, evaluated
    #with parametrs pars.  use dpar for the sampling width for the derivative, and
    #do this for n points 
    grad=np.zeros([n,len(pars)])
    for i in range(len(pars)):
        pp=pars.copy()
        pp[i]=pars[i]+dpar[i]
        y1=fun(pp,n)
        pp[i]=pars[i]-dpar[i]
        y2=fun(pp,n)
        grad[:,i]=(y1-y2)/dpar[i]
    return grad

def newton(pars,fun,dat,noise,dpar,niter=20,fac=1.0):
    #carry out newton's method in multi-dimensions given
    #starting parameters pars, function fun, data dat,
    #noise level noise (assumed white and constant), with
    #width of numerical derivatives set using dpar.
    #if convergence is touchy, you can use fac<1 to not take a full newton step.
    #Rather than try to reason about convergence, just iterate for a fixed number
    #of steps, but the routine does print out chi^2, so it's your responsibility
    #to make sure that has stopped changing.  
    n=len(dat)
    chi0=np.sum(dat**2)/noise**2
    for i in range(niter):
        pred=fun(pars,n)
        r=dat-pred
        grad=numgrad(fun,pars,dpar,n)
        lhs=grad.T@grad
        rhs=grad.T@r
        shift=np.linalg.inv(lhs)@rhs
        chisq=np.sum(r**2)/noise**2
        print('improvement from no model is ',chi0-chisq,pars)
        pars=pars+fac*shift
    return pars
