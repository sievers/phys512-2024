import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def fit_rat(x,y,nump,nq):
    n=len(x)
    assert(nump+nq+1==n)
    A=np.zeros([n,n])
    for i in range(nump+1):
        A[:,i]=x**i
    for i in range(nq):
        A[:,nump+1+i]=-y*x**(i+1)
    fitp=np.linalg.inv(A)@y
    return fitp

def rat_eval(fitp,x,nump,nq):
    top=0
    for i in range(nump+1):
        top=top+x**i*fitp[i]
    bot=1
    for i in range(nump+1,len(fitp)):
        ii=i-nump
        bot=bot+x**ii*fitp[i]
    return top/bot


nump=3
nq=5
x=np.linspace(-1,1,nump+nq+1)
y=np.exp(-0.5*x**2)
fit=fit_rat(x,y,nump,nq)
assert(1==0)
xx=np.linspace(-15,15,1001)
yy=rat_eval(fit,xx,nump,nq)
yy_true=np.exp(-0.5*xx**2)

#pfit=np.polyfit(x,y,10)
#yyp=np.polyval(pfit,xx)

plt.clf()
plt.plot(xx,yy)
plt.plot(x,y,'.')
plt.plot(xx,yy_true)
#plt.plot(xx,yyp)
plt.show()
