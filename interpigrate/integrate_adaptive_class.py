import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def lorentz(x):
    return 1/(1+x**2)

def off_spike(x,sig=0.1):
    y=1+np.exp(-0.5*x**2/sig**2)/sig
    return y
def integrate_adaptive(fun,x0,x1,tol):
    x=np.linspace(x0,x1,5)    
    y=fun(x)
    plt.plot(x,y,'b.')
    dx=x[1]-x[0]

    ans1=(y[0]+4*y[2]+y[4])/6*(x1-x0)
    ans2=(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12*(x1-x0)
    if np.abs(ans1-ans2)<tol:  #we are happy now
        return ans2
    else:
        xmid=(x0+x1)/2
        a1=integrate_adaptive(fun,x0,xmid,tol/2)
        a2=integrate_adaptive(fun,xmid,x1,tol/2)
        return a1+a2

plt.clf()
x0=-100
x1=3
tol=1e-4
if False:
    ans=integrate_adaptive(np.exp,x0,x1,tol)
    truth=np.exp(x1)-np.exp(x0)
elif False:
    ans=integrate_adaptive(lorentz,x0,x1,tol)
    truth=np.arctan(x1)-np.arctan(x0)
else:
    ans=integrate_adaptive(off_spike,x0,x1,tol)
    ans2=integrate_adaptive(off_spike,x0,0,tol)
    ans2=ans2+integrate_adaptive(off_spike,0,x1,tol)
    print("ans2 is ",ans2)
    truth=(x1-x0)+np.sqrt(2*np.pi)
plt.show()

print('answer is ',ans,' vs. ',truth,' with error ',ans-truth,' hoped for ',tol)
