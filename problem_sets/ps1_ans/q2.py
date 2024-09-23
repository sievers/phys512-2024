import numpy as  np
eps=1e-16
def num_diff(fun,x0,dx=0.1,ftol=1e-6):
    isok=False
    xvec=np.linspace(-2,2,5)
    maxiter=5
    iter=0
    while isok==False:
        x=x0+xvec*dx
        y=fun(x)
        d1=(y[4]-y[0])/(4*dx)
        d2=(y[3]-y[1])/(2*dx)
        #normally we care about fractional tolerance.  Here use
        #the average of our neighborhood in case the function is
        #hitting zero at the point we're asking for
        f0=np.mean(np.abs(y))+eps
        tol=ftol*f0
        if np.abs(d2-d1)<tol:
            return (4*d2-d1)/3
        else:
            xmean=np.mean(np.abs(x))
            fpp=np.abs(2*y[2]-y[1]-y[3])/dx**2
            dx=np.sqrt(eps*xmean*f0/fpp)
            iter=iter+1
            if iter==maxiter:
                print('failed to converge in numdiff')
                return (4*d2-d1)/3


            
                        
        
    
